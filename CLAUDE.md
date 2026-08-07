# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

A local agentic RAG system over an aircraft maintenance **cluster** knowledge base. A planner decomposes each question into an evidence plan, several retrieval tools run per question, evidence accumulates with provenance, and a reflection judge loops until the evidence is sufficient.

Every LLM call goes to a local Ollama model (`qwen3:8b`); embeddings are local HuggingFace sentence-transformers. **There is no cloud LLM or embedding provider anywhere in this project — do not introduce one.** The single network dependency is optional Tavily research, which is tightly contained (see below).

Domain vocabulary: a **cluster** is a maintenance-issue category with an ID like `c_38`; each cluster belongs to one of 17 **parent groups** (`baffle`, `engine_general`, `oil_system`, `flight_control`, …). All 51 clusters and their groups come from `data/cluster_taxonomy.csv`.

## Commands

```bash
source .gpt_rags/bin/activate       # project venv (Python 3.14)
pip install -r requirements.txt
ollama serve && ollama pull qwen3:8b

rm -rf chroma_db && python ingest.py  # rebuild the index (see append warning below)
python main.py                        # interactive CLI chat loop
python test_workflow.py               # workflow smoke checks
python test_csv_lookup.py             # CSV layer smoke checks
```

`ingest.py` takes `--data-dir --db-dir --chunk-size --chunk-overlap --embedding-model`. It uses `Chroma.from_documents`, which **appends** — always `rm -rf chroma_db` first or every document is indexed twice.

No test framework, linter, or formatter is configured. Both `test_*.py` files are plain print/assert scripts; run them directly. `test_workflow.py` splits into offline checks (always run) and live checks (need Ollama).

## Architecture

```
main.py → graph.py (LangGraph StateGraph) → nodes.py (thin nodes)
              ↓
   planner.py · research.py · tools.py · taxonomy.py · llm.py
              ↓
   csv_lookup.py (structured) · retriever.py (vector)
```

Flow:

```
triage ──needs research?──yes──> research ─┐
   └──────────────────────no──────────────┴─> build_plan
build_plan → execute_tools → accumulate → reflect
                  ↑                          ├─SUFFICIENT──> generate_answer → END
                  └── replan (gaps) ─────────┴─INSUFFICIENT & budget left
                                              └─budget spent → generate_answer(partial) → END
```

`nodes.py` holds no logic beyond state wiring — planning is in `planner.py`, retrieval in `tools.py`, research in `research.py`.

### Rules the design depends on

- **The planner is a constrained selector, not a generator.** It picks a `tool` from a `Literal` enum and *names* entities; it never writes a retrieval query string — `tools.build_query` does. Every entity it names is validated against the taxonomy by `Planner.validate_steps`, so a hallucinated `c_99` is dropped before reaching a tool. Keep it this way; it is what makes an 8b model safe to plan with.
- **Cluster IDs must be *grounded*, not merely real.** `Planner.validate_steps` drops IDs absent from `grounding_text` (the question plus evidence so far). Existence alone is not enough: asked about "engine clusters" the model names c_12–c_15, which exist but belong to `cowling`/`cylinder_Exhaust`, and scoping a search to them silently searches the wrong documents. When a step loses all its IDs it is **rescoped** to the question's parent groups rather than left unscoped — losing the scope is a worse failure than the wrong scope.
- **A deterministic plan is always the floor.** `Planner.deterministic_plan` runs first, built from keyword signals and taxonomy matching. If the LLM errors or returns nothing valid, that plan is used. The workflow can therefore never be worse than pure keyword routing.
- **Tavily is planning-only and never citable.** `research.py` documents the full contract. The guarantee is structural, not conventional: `nodes.build_answer_prompt` and `build_reflection_prompt` are module-level functions taking only `(question, evidence)`, so `state["research_notes"]` has no path into them. `test_workflow.py` asserts this. Do not add a `research_notes` parameter to either.
- **Metadata filtering is what makes exact cluster IDs safe in vector search.** A search scoped to `cluster: c_38` physically cannot return `c_37`'s document. The old design had to keep exact IDs away from vector retrieval entirely; that restriction is now obsolete, but the *reason* for it still matters — never relax the scoping.
- **Anti-substitution rules** live in both the reflection and answer prompts: never call one cluster "likely" another, never substitute a similar cluster. `nodes.reflect` also enforces this deterministically *before* the LLM — if the question names cluster IDs absent from the evidence, it returns INSUFFICIENT without paying for a call.
- **A nonexistent cluster ID is terminal.** `taxonomy.mentioned_clusters` (unfiltered) and `taxonomy.match_clusters` (taxonomy-filtered) exist as a pair so `reflect` can tell "no cluster was asked about" from "a cluster that does not exist was asked about". The latter sets `state["terminal"]`, which short-circuits the loop — otherwise the validator strips the bad ID, the searches run *unscoped*, and the judge is handed a dozen unrelated clusters to reason over.
- **The loop is bounded** by `config.max_iterations` (default 2 replans = 3 passes). On exhaustion `generate_answer` runs in partial mode: answer what the evidence supports, then state what is missing. `executed_steps` keys prevent a replan from repeating an identical retrieval.
- **`taxonomy.py` is the single source of truth** for cluster IDs, groups, and descriptions. The hardcoded `c_0`–`c_50` map that used to live in `nodes.expand_cluster_ids` is gone — query expansion terms are now generated from the CSV. Don't reintroduce a hardcoded copy.

### Known data gaps

`c_38` (sniffler valve) and `c_50` (spoiler) exist in the taxonomy but have **zero documents in `data/` and zero rows in `final_joined_output.csv`**. Their only evidence is the taxonomy description, so questions about them correctly return INSUFFICIENT for anything beyond the definition. This is a data gap, not a bug — the deterministic comparison plan emits a `metadata_search` step first precisely so compared clusters are never silently empty.

`data/` also carries duplicate coverage for some clusters (`cluster_c_10.txt` and `cluster_c10_baffle_spring.txt`), which is why `nodes.accumulate` deduplicates on `(source, sha1(content[:200]))`.

### Config

`config.py` `RAGConfig` is a dataclass reading env vars (`.env`, loaded with `override=True`). It intentionally carries **three aliases for the same Chroma directory** (`db_dir`, `chroma_persist_directory`, `persist_directory`) plus `data_directory`/`data_dir`, because different modules were written against different names. Keep them consistent if you touch it.

`llm.build_llm` sets `reasoning=False` — qwen3's thinking mode otherwise emits `<think>` blocks that corrupt structured-output parsing and roughly triple latency on what is mostly classification work.

`retriever.py` guards against OpenAI embedding-model names leaking into `HuggingFaceEmbeddings` via `normalize_embedding_model()`. The embedding model at query time **must** match the one used at ingest, or retrieval silently returns garbage. Both `ingest.py` and `retriever.py` pass `collection_name=config.collection_name` — they must agree or retrieval reads an empty collection.

`csv_lookup._normalize_columns()` maps many source column spellings onto canonical names (`workorder`, `ata_code`, `problem`, `action`, `cluster`, `cluster_name`, `parent`, `description`, `registration`, `date*`); the rest of that class assumes only those names. `csv_lookup.py` is reused unchanged as the structured tool — its `_extract_*` staticmethods are called directly from `planner.py`.

## data/

Generated artifacts, mostly. `cluster_<id>.txt` = one RAG document per cluster (what gets embedded, tagged at ingest with `cluster`/`parent`/`doc_type`); `cluster_<group>_unspecified.txt` = whole-group documents (`doc_type=parent_group`); `final_joined_output.csv` = record-level source of truth; `cluster_taxonomy.csv` = cluster→parent→description. `data/README.md` documents regeneration. `chroma_db/` is rebuildable from `data/`.

Note `data/evaluation_questions.json` is stale: its first 8 entries reference tutorial files (`agentic_rag_overview.txt` etc.) that were deleted from `data/`. Only the `cluster_*` entries are meaningful.
