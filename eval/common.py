"""
eval/common.py

Shared pieces for the evaluation scripts: loading the gold set, building the
retrieval stack without an LLM, simulating a gap-driven replan, scoring, and
writing result tables.

Run every eval script from the repo root, e.g. `python eval/eval_retrieval.py`.
Paths in config.py are relative to the repo root; this module also chdirs
there so the scripts work from anywhere.
"""

from __future__ import annotations

import json
import os
import re
import statistics
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence

ROOT = Path(__file__).resolve().parent.parent
os.chdir(ROOT)

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config import RAGConfig  # noqa: E402
from csv_lookup import CSVKnowledgeBase  # noqa: E402
from evidence import EvidenceRanker, merge_pool, select_arrival_order  # noqa: E402
from graph_state import EvidenceItem  # noqa: E402
from planner import Planner, _make_step, step_key  # noqa: E402
from taxonomy import Taxonomy, load_taxonomy  # noqa: E402
from tools import ToolExecutor  # noqa: E402

GOLD_PATH = ROOT / "eval" / "gold_questions.json"
RESULTS_DIR = ROOT / "eval" / "results"


# ----------------------------------------------------------------------
# Gold set
# ----------------------------------------------------------------------


def load_gold(types: Optional[Sequence[str]] = None) -> List[dict]:
    questions = json.loads(GOLD_PATH.read_text())["questions"]

    for question in questions:
        question.setdefault("expected_clusters", [])
        question.setdefault("expected_parents", [])
        question.setdefault("expected_numbers", [])

    if types:
        questions = [q for q in questions if q["type"] in set(types)]

    return questions


# ----------------------------------------------------------------------
# Retrieval stack without an LLM
# ----------------------------------------------------------------------


@dataclass
class Stack:
    config: RAGConfig
    taxonomy: Taxonomy
    knowledge_base: CSVKnowledgeBase
    retriever: object
    executor: ToolExecutor
    planner: Planner
    ranker: EvidenceRanker


def build_stack(config: Optional[RAGConfig] = None) -> Stack:
    """
    Everything the graph uses for retrieval, minus the LLM.

    The planner gets llm=None: only its deterministic plan is used here, so
    retrieval results are reproducible and need no Ollama.
    """
    from retriever import VectorRetriever  # heavy import, deferred

    config = config or RAGConfig()
    taxonomy = load_taxonomy(config.cluster_taxonomy_path)
    knowledge_base = CSVKnowledgeBase(config)
    retriever = VectorRetriever(config)
    embeddings = getattr(retriever, "embeddings", None)

    return Stack(
        config=config,
        taxonomy=taxonomy,
        knowledge_base=knowledge_base,
        retriever=retriever,
        executor=ToolExecutor(config, taxonomy, knowledge_base, retriever),
        planner=Planner(config, llm=None, taxonomy=taxonomy),
        ranker=EvidenceRanker(
            embed_query=getattr(embeddings, "embed_query", None),
            embed_documents=getattr(embeddings, "embed_documents", None),
            rrf_k=config.rrf_k,
        ),
    )


def select(
    stack: Stack,
    question: str,
    pool: Sequence[EvidenceItem],
    selection: str,
) -> List[EvidenceItem]:
    budget = stack.config.max_evidence_chars

    if selection == "arrival":
        return select_arrival_order(pool, budget)

    return stack.ranker.select(
        question=question,
        items=pool,
        budget=budget,
        asked_clusters=stack.taxonomy.match_clusters(question),
        asked_parents=stack.taxonomy.match_parents(question),
    )


def run_retrieval(
    stack: Stack,
    question: str,
    mode: str,
    selection: str,
) -> Dict[str, object]:
    """
    Two LLM-free passes that mirror the graph.

    Pass 1 runs the deterministic plan. Pass 2 is the replan the reflect node
    would trigger: for every cluster the QUESTION names (never the gold
    labels) that has no substantive evidence in the selection, run a search
    scoped to it with the gap text reflect would emit. Evidence then goes
    through the same selection the graph uses.
    """
    stack.retriever.mode = mode
    started = time.perf_counter()

    plan = stack.planner.deterministic_plan(question)
    executed = {step_key(step) for step in plan}
    pending: List[EvidenceItem] = []

    for step in plan:
        pending.extend(stack.executor.execute(step))

    pool = merge_pool([], pending)
    selected = select(stack, question, pool, selection)
    passes = 1

    asked = stack.taxonomy.match_clusters(question)
    missing = [
        cluster
        for cluster in asked
        if not any(
            item["cluster"] == cluster and item["tool"] != "metadata_search"
            for item in selected
        )
    ]

    gap_steps = [
        _make_step("vector_search", f"Evidence for cluster {cluster}", cluster_ids=[cluster])
        for cluster in missing
    ]
    gap_steps = [step for step in gap_steps if step_key(step) not in executed]

    if gap_steps:
        passes = 2
        new_items: List[EvidenceItem] = []

        for step in gap_steps:
            new_items.extend(stack.executor.execute(step))

        pool = merge_pool(pool, new_items)
        selected = select(stack, question, pool, selection)

    return {
        "evidence": selected,
        "pool_size": len(pool),
        "passes": passes,
        "seconds": time.perf_counter() - started,
    }


# ----------------------------------------------------------------------
# Scoring
# ----------------------------------------------------------------------


def _relevant_parents(gold: dict, taxonomy: Taxonomy) -> set:
    parents = set(gold["expected_parents"])
    parents.update(taxonomy.parent_of(c) for c in gold["expected_clusters"])
    parents.discard("")
    return parents


def is_exact_match(item: EvidenceItem, gold: dict) -> bool:
    """
    Evidence about exactly what was asked: the named cluster, or for a group
    question any item tagged with that group.
    """
    if gold["expected_clusters"]:
        return item["cluster"] in gold["expected_clusters"]

    return item["parent"] in gold["expected_parents"]


def score_evidence(
    gold: dict,
    evidence: Sequence[EvidenceItem],
    taxonomy: Taxonomy,
) -> Dict[str, Optional[float]]:
    """
    Per-question retrieval metrics.

    coverage         share of expected clusters/groups with at least one
                     substantive (non-definition) item. For c_38/c_50, which
                     have no documents, a definition counts.
    balanced         compare questions only: 1 if every side has substantive
                     evidence, else 0.
    mrr              1 / rank of the first exact-match item in the selection.
    scope_precision  of the items tagged with any cluster or group, the share
                     tagged with an expected cluster or its group.
    fact_recall      share of expected numbers present in the evidence text.
    """
    entities = gold["expected_clusters"] or gold["expected_parents"]
    field = "cluster" if gold["expected_clusters"] else "parent"
    no_docs = gold["type"] == "no_records"

    def has_substantive(value: str) -> bool:
        return any(
            item[field] == value and (no_docs or item["tool"] != "metadata_search")
            for item in evidence
        )

    covered = [has_substantive(value) for value in entities]
    coverage = sum(covered) / len(covered) if covered else None

    rank = next(
        (i for i, item in enumerate(evidence, start=1) if is_exact_match(item, gold)),
        None,
    )

    relevant_parents = _relevant_parents(gold, taxonomy)
    tagged = [item for item in evidence if item["cluster"] or item["parent"]]
    in_scope = [
        item
        for item in tagged
        if item["cluster"] in gold["expected_clusters"]
        or (not item["cluster"] and item["parent"] in relevant_parents)
        or (item["cluster"] and not gold["expected_clusters"] and item["parent"] in relevant_parents)
    ]

    text = normalize_numbers(" ".join(item["content"] for item in evidence))
    numbers = gold["expected_numbers"]

    return {
        "coverage": coverage,
        "balanced": (1.0 if all(covered) else 0.0) if gold["type"] == "compare" else None,
        "mrr": (1.0 / rank) if rank else 0.0,
        "scope_precision": (len(in_scope) / len(tagged)) if tagged else None,
        "fact_recall": (
            sum(1 for n in numbers if re.search(rf"\b{re.escape(n)}\b", text)) / len(numbers)
            if numbers
            else None
        ),
        "items": float(len(evidence)),
        "chars": float(sum(len(item["content"]) for item in evidence)),
    }


def normalize_numbers(text: str) -> str:
    """
    7,884 -> 7884 and 7230.0 -> 7230, so number checks are format-blind.
    """
    text = re.sub(r"(?<=\d),(?=\d{3}\b)", "", text)
    return re.sub(r"\b(\d+)\.0\b", r"\1", text)


def mean_or_none(values: Sequence[Optional[float]]) -> Optional[float]:
    present = [v for v in values if v is not None]
    return statistics.mean(present) if present else None


def summarize(rows: Sequence[dict], metrics: Sequence[str]) -> Dict[str, Optional[float]]:
    return {metric: mean_or_none([row["metrics"].get(metric) for row in rows]) for metric in metrics}


def percentile(values: Sequence[float], fraction: float) -> float:
    if not values:
        return 0.0

    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, int(round(fraction * (len(ordered) - 1))))]


# ----------------------------------------------------------------------
# Output
# ----------------------------------------------------------------------


def fmt(value: Optional[float], digits: int = 3) -> str:
    if value is None:
        return "-"

    if digits == 0:
        return f"{value:.0f}"

    return f"{value:.{digits}f}"


def markdown_table(headers: Sequence[str], rows: Sequence[Sequence[str]]) -> str:
    lines = [
        "| " + " | ".join(headers) + " |",
        "|" + "|".join("---" for _ in headers) + "|",
    ]
    lines += ["| " + " | ".join(row) + " |" for row in rows]
    return "\n".join(lines)


def save_results(name: str, payload: dict, report: str) -> Path:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y%m%d_%H%M%S")
    json_path = RESULTS_DIR / f"{name}_{stamp}.json"
    json_path.write_text(json.dumps(payload, indent=2, default=str))
    (RESULTS_DIR / f"{name}_{stamp}.md").write_text(report)
    return json_path
