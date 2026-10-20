"""
test_retrieval.py

Offline checks for BM25, rank fusion, chunking and evidence selection.
Plain script, like the other test files. Needs only pandas (no Ollama, no
Chroma, no embedding model).

Run: python test_retrieval.py
"""

from bm25 import BM25, reciprocal_rank_fusion, tokenize
from chunking import split_markdown
from evidence import EvidenceRanker, merge_pool, select_arrival_order
from graph_state import EvidenceItem
from taxonomy import load_taxonomy


PASSED = []
FAILED = []


def check(name: str, condition: bool, detail: str = "") -> None:
    if condition:
        PASSED.append(name)
        print(f"PASS  {name}")
    else:
        FAILED.append(name)
        print(f"FAIL  {name} {detail}")


def item(content: str, cluster: str = "", parent: str = "", tool: str = "vector_search",
         source: str = "data/x.txt") -> EvidenceItem:
    return {
        "tool": tool,
        "content": content,
        "source": source,
        "cluster": cluster,
        "parent": parent,
    }


# --- Tokenizer ------------------------------------------------------------

tokens = tokenize("L/H MAG drop on A/C, ATA 7230.0, cluster c22, R&R plugs")

check("shorthand is expanded", {"left", "magneto", "aircraft", "replace", "plug"} <= set(tokens), str(tokens))
check("ATA codes lose the pandas .0", "7230" in tokens, str(tokens))
check("c22 is normalized to c_22", "c_22" in tokens, str(tokens))
check("stopwords are dropped", "on" not in tokens and "the" not in tokenize("the baffle"))

# --- BM25 -----------------------------------------------------------------

corpus = [
    "magneto mag drop on runup left mag rough",
    "spark plug fouled lead deposits cleaned plugs",
    "baffle crack repaired with doubler",
]
index = BM25([tokenize(text) for text in corpus])

check("BM25 ranks the exact-term document first", index.top_n("mag drop", 3)[0][0] == 0)
check("BM25 honours the allowed scope", all(i != 0 for i, _ in index.top_n("mag drop", 3, allowed=[1, 2])))
check("BM25 never returns zero-score documents", index.top_n("propeller overspeed", 3) == [])

fused = reciprocal_rank_fusion([["a", "b", "c"], ["c", "a", "d"]])
check("RRF rewards agreement between rankers", fused[0][0] == "a", str(fused))
check("RRF keeps items seen by only one ranker", {"b", "d"} <= {key for key, _ in fused})

# --- Chunking -------------------------------------------------------------

with open("data/cluster_c_22.txt", encoding="utf-8") as handle:
    document = handle.read()

chunks = split_markdown(document, chunk_size=600, chunk_overlap=100)

check("no chunk exceeds chunk_size", all(len(text) <= 600 for text, _ in chunks),
      str(max(len(text) for text, _ in chunks)))
check("every chunk names its cluster", all(text.startswith("Cluster c_22") for text, _ in chunks))
check("example records are never split",
      all(text.count("Work order:") == text.count("Action:") for text, _ in chunks))
check("small neighbouring sections are packed together",
      any("," in meta["section"] for _, meta in chunks), str([m["section"] for _, m in chunks]))
check("no text is lost", all(
    line.strip() in "\n".join(text for text, _ in chunks)
    for line in document.splitlines() if line.strip() and not line.startswith("# ")
))

long_section = "# T\n\n## Terms\n" + "\n".join(f"- term{i}: {i}" for i in range(200))
long_chunks = split_markdown(long_section, chunk_size=300, chunk_overlap=60)
check("oversized sections are cut under the limit", all(len(t) <= 300 for t, _ in long_chunks))
check("cuts carry overlapping lines", any(
    a.splitlines()[-1] in b for (a, _), (b, _) in zip(long_chunks, long_chunks[1:])
))

# --- Evidence selection ---------------------------------------------------

ranker = EvidenceRanker()  # BM25-only ranking; embeddings are optional

# The old failure: pass 1 fills the budget with weak chunks, then the replan's
# exact evidence arrives and is dropped.
weak = [item(f"cowling paint wash cosmetic note {i} " * 12, cluster="c_43") for i in range(4)]
strong = item("c_30 magneto mag drop: left mag drop 150 rpm, replaced magneto", cluster="c_30")
pool = merge_pool(weak, [strong])
budget = sum(len(x["content"]) for x in weak)

check("arrival order drops the late, relevant item (the old bug)",
      strong not in select_arrival_order(pool, budget))
check("ranked selection keeps the late, relevant item",
      strong in ranker.select("What causes mag drop in cluster c_30?", pool, budget, asked_clusters=["c_30"]))
check("ranked selection puts the relevant item first",
      ranker.select("What causes mag drop in cluster c_30?", pool, budget, asked_clusters=["c_30"])[0] is strong)

# Comparisons: one side must not take the whole budget.
side_a = [item(f"c_30 magneto chunk {i} " * 20, cluster="c_30") for i in range(6)]
side_b = [item(f"c_39 spark plug chunk {i} " * 20, cluster="c_39") for i in range(6)]
picked = ranker.select("Compare c_30 and c_39 magneto", side_a + side_b, 2400,
                       asked_clusters=["c_30", "c_39"])
check("comparison keeps evidence for both sides", {"c_30", "c_39"} <= {x["cluster"] for x in picked})
check("comparison shares the budget fairly",
      abs(sum(x["cluster"] == "c_30" for x in picked) - sum(x["cluster"] == "c_39" for x in picked)) <= 1,
      str([x["cluster"] for x in picked]))

# A definition is acceptable coverage when it is all that exists (c_38).
definition = item("c_38: sniffler valve", cluster="c_38", tool="metadata_search", source="data/cluster_taxonomy.csv")
picked = ranker.select("What is cluster c_38?", weak + [definition], budget, asked_clusters=["c_38"])
check("a lone definition still covers its cluster", definition in picked)

# Exact CSV facts are pinned for structured questions.
fact = item("Number of records in c_22: 1609", cluster="c_22", tool="csv_lookup", source="data/final_joined_output.csv")
noise = [item("c_22 idle rpm mixture records records records " * 10, cluster="c_22") for _ in range(3)]
picked = ranker.select("How many records are in cluster c_22?", noise + [fact], 1200, asked_clusters=["c_22"])
check("CSV facts are pinned first for count questions", picked[0] is fact)

check("budget is never exceeded", sum(len(x["content"]) for x in picked) <= 1200)

check("duplicates are merged once", len(merge_pool([fact], [dict(fact)])) == 1)

# --- Cluster-ID matching (the c_1 / c_10 bug) -----------------------------

taxonomy = load_taxonomy("data/cluster_taxonomy.csv")
check("c_1 is not found inside c_10", "c_1" not in taxonomy.mentioned_clusters("evidence about c_10 only"))

# --- Summary --------------------------------------------------------------

print()
print("=" * 70)
print(f"{len(PASSED)} passed, {len(FAILED)} failed")

if FAILED:
    for name in FAILED:
        print(f"  failed: {name}")
    raise SystemExit(1)
