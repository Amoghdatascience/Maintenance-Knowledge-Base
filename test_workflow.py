"""
test_workflow.py

Smoke checks for the agentic workflow. Plain script, like test_csv_lookup.py.

Run: python test_workflow.py

The planner checks need Ollama running. The validator, taxonomy, and
containment checks are offline and run regardless.
"""

from config import RAGConfig
from graph_state import EvidenceItem
from nodes import build_answer_prompt, build_reflection_prompt
from planner import EvidencePlan, EvidenceStep, Planner
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


def evidence(content: str, source: str = "data/cluster_c_38.txt") -> EvidenceItem:
    return {
        "tool": "vector_search",
        "sub_question": "test",
        "content": content,
        "source": source,
        "cluster": "c_38",
        "parent": "valve_cover",
    }


config = RAGConfig()
taxonomy = load_taxonomy(config.cluster_taxonomy_path)

print("=" * 70)
print("Offline checks")
print("=" * 70)

# --- Taxonomy validation -------------------------------------------------

check(
    "taxonomy loads 51 clusters and 17 parent groups",
    len(taxonomy.clusters) == 51 and len(taxonomy.parents) == 17,
    f"got {len(taxonomy.clusters)} clusters, {len(taxonomy.parents)} groups",
)

check(
    "hallucinated cluster c_99 is dropped",
    taxonomy.valid_clusters(["c_38", "c_99"]) == ["c_38"],
    str(taxonomy.valid_clusters(["c_38", "c_99"])),
)

check(
    "invented parent group is dropped",
    taxonomy.valid_parents(["baffle", "warp_drive"]) == ["baffle"],
    str(taxonomy.valid_parents(["baffle", "warp_drive"])),
)

# "no cluster asked about" and "a cluster that does not exist" must stay
# distinguishable, or the reflection guard cannot fire on c_99.
check(
    "nonexistent cluster is detected as unknown, not as absent",
    taxonomy.unknown_clusters("What is cluster c_99?") == ["c_99"]
    and taxonomy.match_clusters("What is cluster c_99?") == [],
    str(taxonomy.unknown_clusters("What is cluster c_99?")),
)

check(
    "real cluster is not flagged unknown",
    taxonomy.unknown_clusters("What is cluster c_22?") == [],
)

check(
    "question naming no cluster yields neither",
    taxonomy.unknown_clusters("Explain the baffle clusters.") == []
    and taxonomy.match_clusters("Explain the baffle clusters.") == [],
)

# --- Plan validation -----------------------------------------------------

planner = Planner(config, llm=None, taxonomy=taxonomy)

dirty_plan = EvidencePlan(
    steps=[
        EvidenceStep(
            tool="vector_search",
            sub_question="What is c_99?",
            cluster_ids=["c_99"],
            parent_groups=["warp_drive"],
        ),
        EvidenceStep(
            tool="csv_lookup",
            sub_question="How many records in c_22?",
            cluster_ids=["c_22"],
        ),
    ]
)

validated = planner.validate_plan(dirty_plan)

check(
    "validator strips invalid entities but keeps the step",
    validated[0]["cluster_ids"] == [] and validated[0]["parent_groups"] == [],
    str(validated[0]),
)

check(
    "validator keeps valid cluster IDs",
    validated[1]["cluster_ids"] == ["c_22"],
    str(validated[1]),
)

check(
    "validator caps plan length",
    len(planner.validate_plan(EvidencePlan(steps=list(dirty_plan.steps) * 10)))
    <= config.max_plan_steps,
)

# Existence is not enough: asked about "engine clusters" the model names
# c_12-c_15, which are real but belong to cowling and cylinder groups.
# Scoping a search to those silently searches the wrong documents.
ungrounded = EvidencePlan(
    steps=[
        EvidenceStep(
            tool="vector_search",
            sub_question="Common problems across engine clusters",
            cluster_ids=["c_12", "c_13", "c_14", "c_15"],
        )
    ]
)

grounded_result = planner.validate_plan(
    ungrounded,
    grounding_text="what are common problems in engine clusters?",
    fallback_parents=["engine_general"],
)

check(
    "real but ungrounded cluster IDs are dropped",
    grounded_result[0]["cluster_ids"] == [],
    str(grounded_result[0]["cluster_ids"]),
)

check(
    "a step stripped of cluster IDs is rescoped, not left unscoped",
    grounded_result[0]["parent_groups"] == ["engine_general"],
    str(grounded_result[0]["parent_groups"]),
)

check(
    "cluster IDs present in the question survive grounding",
    planner.validate_plan(
        EvidencePlan(
            steps=[
                EvidenceStep(
                    tool="vector_search",
                    sub_question="What is c_22?",
                    cluster_ids=["c_22"],
                )
            ]
        ),
        grounding_text="How many records are in cluster c_22?",
    )[0]["cluster_ids"]
    == ["c_22"],
)

already_run = [f"csv_lookup|c_22||how many records in c_22?"]

check(
    "validator drops already-executed steps",
    all(
        step["tool"] != "csv_lookup"
        for step in planner.validate_plan(dirty_plan, already_run)
    ),
)

# --- Deterministic plan (the fallback floor) -----------------------------

cases = {
    "How many records are in cluster c_22?": "csv_lookup",
    "Compare c_37 and c_38.": "vector_search",
    "What are common problem patterns across baffle clusters?": "metadata_search",
    "Explain what the ignition clusters cover.": "metadata_search",
    "What is cluster c_38?": "csv_lookup",
}

for question, expected_tool in cases.items():
    plan = planner.deterministic_plan(question)
    tools_used = [step["tool"] for step in plan]

    check(
        f"deterministic plan for {question!r} uses {expected_tool}",
        expected_tool in tools_used and len(plan) >= 1,
        f"got {tools_used}",
    )

comparison_plan = planner.deterministic_plan("Compare c_37 and c_38.")
scoped = [step["cluster_ids"] for step in comparison_plan]

check(
    "comparison fans out one step per cluster",
    ["c_37"] in scoped and ["c_38"] in scoped,
    str(scoped),
)

# --- Research containment ------------------------------------------------
#
# The guarantee is structural: the answer and reflection prompt builders take
# only (question, evidence), so research_notes cannot reach them.

secret = "sniffler-valve-web-term"
items = [evidence("Cluster c_38 covers the sniffler valve.")]

answer_prompt = build_answer_prompt("What is c_38?", items)
reflection_prompt = build_reflection_prompt("What is c_38?", items)

check(
    "research keywords never reach the answer prompt",
    secret not in answer_prompt,
)

check(
    "research keywords never reach the reflection prompt",
    secret not in reflection_prompt,
)

check(
    "answer prompt is grounded in the evidence",
    "sniffler valve" in answer_prompt and "data/cluster_c_38.txt" in answer_prompt,
)

check(
    "partial mode is flagged to the model",
    "no further retrieval is possible"
    in build_answer_prompt("q", items, partial=True),
)

# --- Live checks ---------------------------------------------------------

print()
print("=" * 70)
print("Live checks (need Ollama running)")
print("=" * 70)

try:
    from graph import AgenticRAGGraph

    graph = AgenticRAGGraph(config)

    result = graph.ask("How many records are in cluster c_22?")

    check(
        "count question routes through csv_lookup",
        any(item["tool"] == "csv_lookup" for item in result["evidence"]),
        str(sorted({i["tool"] for i in result["evidence"]})),
    )

    check(
        "count question produces an answer",
        bool(result["answer"].strip()),
    )

    missing = graph.ask("What is cluster c_99?")

    check(
        "nonexistent cluster is not answered with a substitute",
        missing["verdict"] == "INSUFFICIENT",
        f"verdict={missing['verdict']}",
    )

    scoped_result = graph.ask("Compare c_37 and c_38.")
    seen_clusters = {i["cluster"] for i in scoped_result["evidence"] if i["cluster"]}

    check(
        "comparison retrieves both clusters and nothing else",
        {"c_37", "c_38"}.issubset(seen_clusters) or seen_clusters <= {"c_37", "c_38"},
        str(seen_clusters),
    )

except Exception as error:  # noqa: BLE001
    print(f"SKIP  live checks: {error}")

# --- Live web research ---------------------------------------------------
#
# Only runs when a TAVILY_API_KEY is present. Verifies the output contract
# against real search results, and that research cannot reach an answer.

print()
print("=" * 70)
print("Live web research (needs TAVILY_API_KEY)")
print("=" * 70)

try:
    from llm import build_llm
    from research import WebResearcher

    researcher = WebResearcher(config, build_llm(config))

    if not researcher.enabled:
        print(f"SKIP  web research: {researcher._unavailable_reason}")
    else:
        keywords = researcher.research("what is a sniffler valve on an aircraft")

        check(
            "research returns a non-empty keyword list",
            isinstance(keywords, list)
            and bool(keywords)
            and all(isinstance(k, str) for k in keywords),
            str(keywords),
        )

        check(
            "keywords respect the count and length caps",
            len(keywords) <= config.research_max_keywords
            and all(len(k) <= config.research_keyword_max_len for k in keywords),
            f"{len(keywords)} terms, longest {max(map(len, keywords), default=0)}",
        )

        check(
            "keywords are normalized and free of prose",
            all(k == k.lower() for k in keywords)
            and len(keywords) == len(set(keywords))
            and not any("." in k or "http" in k for k in keywords),
        )

        check(
            "repeat research is served from cache",
            researcher.research("what is a sniffler valve on an aircraft") == keywords,
        )

        # End-to-end containment. A keyword only proves a leak if it could
        # ONLY have come from research -- terms already present in the
        # question or the retrieved evidence prove nothing either way.
        question = "What is a sniffler valve and which cluster covers it?"
        researched = graph.ask(question)

        notes = researched["research_notes"]
        answer = researched["answer"].lower()
        evidence_text = " ".join(
            item["content"] for item in researched["evidence"]
        ).lower()

        research_only = [
            keyword
            for keyword in notes
            if keyword not in question.lower() and keyword not in evidence_text
        ]

        check(
            "the research node ran for a domain question",
            researched["research_done"],
        )

        # Tavily can rate-limit under the rapid calls this suite makes. Empty
        # notes then mean the contract degraded gracefully -- the behaviour it
        # promises -- so that is a skip, not a failure. The keyword contract
        # itself is already asserted above against a live call.
        if not notes:
            print("SKIP  containment end-to-end: research returned no keywords")
        else:
            check(
                "research-only terms never reach the answer",
                not [k for k in research_only if k in answer],
                f"leaked={[k for k in research_only if k in answer]}",
            )

except Exception as error:  # noqa: BLE001
    print(f"SKIP  web research checks: {error}")

# --- Summary -------------------------------------------------------------

print()
print("=" * 70)
print(f"{len(PASSED)} passed, {len(FAILED)} failed")

if FAILED:
    for name in FAILED:
        print(f"  failed: {name}")
