"""
eval/eval_e2e.py

End-to-end evaluation of the full agentic graph. Needs `ollama serve`.

For every gold question it runs graph.ask() and checks the final answer:

    numbers_correct      every expected number appears in the answer
    refusal_correct      nonexistent clusters are refused, never substituted
    no_records_correct   c_38 / c_50 answers say there is no data
    ungrounded_numbers   answer numbers (2+ digits) absent from the evidence
                         and the question -- the hallucination check
    ungrounded_clusters  answer cluster IDs absent from evidence and question
    cites_source         the answer names a source file from its evidence
    coverage             retrieval coverage of the evidence the answer used

plus latency (p50 / p95), retrieval passes and verdicts.

At ~15-40 s per question on a laptop, the full set takes 15-25 minutes.
Use --limit or --types for a quick run.

Usage (from the repo root):
    python eval/eval_e2e.py
    python eval/eval_e2e.py --types count compare --limit 5
"""

from __future__ import annotations

import argparse
import re
import time
from pathlib import Path
from typing import List, Optional

from common import (
    fmt,
    load_gold,
    markdown_table,
    mean_or_none,
    normalize_numbers,
    percentile,
    save_results,
    score_evidence,
)

from config import RAGConfig
from graph import AgenticRAGGraph
from llm import warm_up

NO_DATA_PHRASES = (
    "no records", "0 records", "zero records", "no document", "no data",
    "insufficient", "not available", "does not contain", "no evidence",
)
REFUSAL_PHRASES = ("not a cluster", "does not exist", "no such cluster", "not in the taxonomy")


def numbers_in(text: str) -> List[str]:
    """Numbers with 2+ digits. Single digits are list markers and noise."""
    return re.findall(r"\b\d{2,}\b", normalize_numbers(text))


def check_answer(gold: dict, result: dict, taxonomy) -> dict:
    answer = result.get("answer", "")
    lowered = answer.lower()
    evidence = result.get("evidence", [])
    support = normalize_numbers(
        gold["question"] + " " + " ".join(item["content"] for item in evidence)
    )
    support_numbers = set(re.findall(r"\b\d+\b", support))
    support_clusters = set(taxonomy.mentioned_clusters(support))
    support_clusters.update(item["cluster"] for item in evidence if item["cluster"])

    # A terminal refusal is written by code, not the model ("valid IDs run
    # c_0 to c_50"), so it has nothing to hallucinate and is not checked.
    checked = "" if result.get("terminal") else answer
    answer_numbers = set(numbers_in(checked))
    all_answer_numbers = set(re.findall(r"\b\d+\b", normalize_numbers(answer)))
    answer_clusters = set(taxonomy.mentioned_clusters(checked))
    sources = {Path(item["source"]).name for item in evidence if item.get("source")}

    expected = gold["expected_numbers"]
    kind = gold["type"]

    metrics: dict = {
        "numbers_correct": (
            float(all(n in all_answer_numbers for n in expected)) if expected else None
        ),
        "refusal_correct": (
            float(bool(result.get("terminal")) or any(p in lowered for p in REFUSAL_PHRASES))
            if kind == "refusal"
            else None
        ),
        "no_records_correct": (
            float(any(p in lowered for p in NO_DATA_PHRASES)) if kind == "no_records" else None
        ),
        "ungrounded_numbers": float(len(answer_numbers - support_numbers)),
        "ungrounded_clusters": float(len(answer_clusters - support_clusters)),
        "cites_source": (
            float(any(source in answer for source in sources) or "final_joined_output" in answer)
            if evidence
            else None
        ),
    }

    if kind != "refusal":
        metrics["coverage"] = score_evidence(gold, evidence, taxonomy)["coverage"]

    metrics["examples"] = {
        "ungrounded_numbers": sorted(answer_numbers - support_numbers),
        "ungrounded_clusters": sorted(answer_clusters - support_clusters),
    }

    return metrics


def rate(rows: List[dict], metric: str) -> Optional[float]:
    return mean_or_none([row["metrics"].get(metric) for row in rows])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    parser.add_argument("--types", nargs="+", default=None)
    parser.add_argument("--ids", nargs="+", default=None)
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    gold = load_gold(args.types)

    if args.ids:
        gold = [q for q in gold if q["id"] in set(args.ids)]

    if args.limit:
        gold = gold[: args.limit]

    config = RAGConfig()
    graph = AgenticRAGGraph(config)
    taxonomy = graph.nodes.taxonomy

    print("Loading model...")
    warm_up(graph.nodes.llm)

    rows = []

    for number, question in enumerate(gold, start=1):
        print(f"\n[{number}/{len(gold)}] {question['question']}")
        started = time.perf_counter()

        try:
            result = graph.ask(question["question"])
            error = ""
        except Exception as exc:  # noqa: BLE001 - one failure must not stop the eval
            result, error = {}, str(exc)

        seconds = time.perf_counter() - started
        metrics = check_answer(question, result, taxonomy) if result else {}

        rows.append(
            {
                "id": question["id"],
                "type": question["type"],
                "seconds": seconds,
                "passes": result.get("iterations", 0),
                "verdict": result.get("verdict", ""),
                "error": error,
                "answer": result.get("answer", ""),
                "metrics": metrics,
            }
        )

        print(f"  {seconds:.1f}s, passes={result.get('iterations', 0)}, verdict={result.get('verdict', '')}")

    seconds = [row["seconds"] for row in rows]
    answered = [row for row in rows if not row["error"]]

    summary = markdown_table(
        ["metric", "value"],
        [
            ["questions", str(len(rows))],
            ["errors", str(len(rows) - len(answered))],
            ["numbers correct (count questions)", fmt(rate(answered, "numbers_correct"))],
            ["refusal correct", fmt(rate(answered, "refusal_correct"))],
            ["no-records correct", fmt(rate(answered, "no_records_correct"))],
            ["answers with ungrounded numbers", fmt(mean_or_none([1.0 if r["metrics"].get("ungrounded_numbers") else 0.0 for r in answered]))],
            ["answers with ungrounded cluster IDs", fmt(mean_or_none([1.0 if r["metrics"].get("ungrounded_clusters") else 0.0 for r in answered]))],
            ["cites a source", fmt(rate(answered, "cites_source"))],
            ["evidence coverage", fmt(rate(answered, "coverage"))],
            ["latency p50 (s)", fmt(percentile(seconds, 0.5), 1)],
            ["latency p95 (s)", fmt(percentile(seconds, 0.95), 1)],
            ["mean retrieval passes", fmt(mean_or_none([float(r["passes"]) for r in answered]), 2)],
        ],
    )

    flagged = [
        f"- {row['id']}: ungrounded numbers {row['metrics']['examples']['ungrounded_numbers']}, "
        f"clusters {row['metrics']['examples']['ungrounded_clusters']}"
        for row in answered
        if row["metrics"].get("ungrounded_numbers") or row["metrics"].get("ungrounded_clusters")
    ]

    report = "\n\n".join(
        [
            "# End-to-end evaluation",
            f"Config: retrieval={config.retrieval_mode}, selection={config.evidence_selection}, "
            f"model={config.ollama_model}",
            summary,
            "## Answers to review (possible hallucinations)",
            "\n".join(flagged) or "None flagged.",
        ]
    )

    print()
    print(report)

    path = save_results("e2e", {"rows": rows}, report)
    print(f"\nSaved {path}")


if __name__ == "__main__":
    main()
