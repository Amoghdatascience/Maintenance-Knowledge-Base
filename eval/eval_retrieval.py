"""
eval/eval_retrieval.py

Retrieval evaluation. No LLM needed, so it is fast and fully reproducible.

Part A, raw search quality: the gold "lookup" questions ("Which cluster
covers mag drop problems?") are searched UNSCOPED with each retrieval mode.
Scored by MRR@k and hit@3 on the first chunk from the expected cluster.
This isolates vector vs BM25 vs hybrid with nothing else in the way.

Part B, the full evidence pipeline: every non-refusal gold question goes
through the deterministic plan, the tools, a simulated gap replan and the
evidence selection, for every combination of

    retrieval mode   vector | bm25 | hybrid
    selection        arrival (old: first in wins) | ranked (new)

"vector + arrival" is the system before this change; "hybrid + ranked" is
the new default. Metrics are defined in eval/common.py:score_evidence.

Usage (from the repo root, after ingest):
    python eval/eval_retrieval.py
    python eval/eval_retrieval.py --modes hybrid --selections ranked arrival
"""

from __future__ import annotations

import argparse
from typing import Dict, List

from common import (
    build_stack,
    fmt,
    load_gold,
    markdown_table,
    mean_or_none,
    run_retrieval,
    save_results,
    score_evidence,
    summarize,
)

METRICS = ["coverage", "balanced", "mrr", "scope_precision", "fact_recall", "items", "chars"]
TYPES = ["count", "explain", "compare", "group", "lookup", "no_records"]


def raw_search(stack, gold: List[dict], modes: List[str], k: int) -> Dict[str, dict]:
    """
    Part A: unscoped search for lookup questions, per retrieval mode.
    """
    lookups = [q for q in gold if q["type"] == "lookup"]
    results: Dict[str, dict] = {}

    for mode in modes:
        rows = []

        for question in lookups:
            documents = stack.retriever.retrieve(question["question"], k=k, mode=mode)
            clusters = [str((doc.metadata or {}).get("cluster", "")) for doc in documents]
            rank = next(
                (i for i, c in enumerate(clusters, start=1) if c in question["expected_clusters"]),
                None,
            )
            rows.append(
                {
                    "id": question["id"],
                    "rank": rank,
                    "top_clusters": clusters[:5],
                }
            )

        results[mode] = {
            "mrr": mean_or_none([1.0 / r["rank"] if r["rank"] else 0.0 for r in rows]),
            "hit@3": mean_or_none([1.0 if r["rank"] and r["rank"] <= 3 else 0.0 for r in rows]),
            "hit@k": mean_or_none([1.0 if r["rank"] else 0.0 for r in rows]),
            "rows": rows,
        }

    return results


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    parser.add_argument("--modes", nargs="+", default=["vector", "bm25", "hybrid"])
    parser.add_argument("--selections", nargs="+", default=["arrival", "ranked"])
    parser.add_argument("--k", type=int, default=8, help="top-k for Part A")
    args = parser.parse_args()

    stack = build_stack()
    gold = [q for q in load_gold() if q["type"] != "refusal"]

    # ------------------------------------------------------------- Part A
    raw = raw_search(stack, gold, args.modes, args.k)

    part_a = markdown_table(
        ["mode", f"MRR@{args.k}", "hit@3", f"hit@{args.k}"],
        [[m, fmt(r["mrr"]), fmt(r["hit@3"]), fmt(r["hit@k"])] for m, r in raw.items()],
    )

    # ------------------------------------------------------------- Part B
    runs: Dict[str, List[dict]] = {}

    for mode in args.modes:
        for selection in args.selections:
            name = f"{mode}+{selection}"
            rows = []

            for question in gold:
                outcome = run_retrieval(stack, question["question"], mode, selection)
                rows.append(
                    {
                        "id": question["id"],
                        "type": question["type"],
                        "passes": outcome["passes"],
                        "pool_size": outcome["pool_size"],
                        "seconds": outcome["seconds"],
                        "metrics": score_evidence(question, outcome["evidence"], stack.taxonomy),
                        "evidence": [
                            {k: item[k] for k in ("tool", "source", "cluster", "parent")}
                            for item in outcome["evidence"]
                        ],
                    }
                )

            runs[name] = rows
            print(f"[eval] finished {name}")

    overall_rows = []

    for name, rows in runs.items():
        s = summarize(rows, METRICS)
        overall_rows.append(
            [
                name,
                fmt(s["coverage"]),
                fmt(s["balanced"]),
                fmt(s["mrr"]),
                fmt(s["scope_precision"]),
                fmt(s["fact_recall"]),
                fmt(s["items"], 1),
                fmt(s["chars"], 0),
            ]
        )

    part_b = markdown_table(
        ["config", "coverage", "balanced (compare)", "MRR", "scope precision", "fact recall", "items", "chars"],
        overall_rows,
    )

    by_type_rows = []

    for name, rows in runs.items():
        cells = [name]

        for kind in TYPES:
            subset = [row for row in rows if row["type"] == kind]
            cells.append(fmt(summarize(subset, ["coverage"])["coverage"]) if subset else "-")

        by_type_rows.append(cells)

    part_b_types = markdown_table(["config", *TYPES], by_type_rows)

    report = "\n\n".join(
        [
            "# Retrieval evaluation",
            f"Gold questions: {len(gold)} (refusals excluded; see eval_e2e.py)",
            "## Part A: raw unscoped search on lookup questions",
            part_a,
            "## Part B: full evidence pipeline (deterministic plan + gap replan + selection)",
            part_b,
            "### Coverage by question type",
            part_b_types,
            "Baseline (old system) = vector+arrival. New default = hybrid+ranked.",
        ]
    )

    print()
    print(report)

    path = save_results("retrieval", {"raw_search": raw, "runs": runs}, report)
    print(f"\nSaved {path}")


if __name__ == "__main__":
    main()
