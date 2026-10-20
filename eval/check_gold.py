"""
eval/check_gold.py

Validates the gold set against the data, so the eval can never silently
score against stale answers:

- every expected cluster / parent exists in the taxonomy
- every count question's expected number matches final_joined_output.csv
- refusal questions really name a cluster that does not exist

Usage (from the repo root): python eval/check_gold.py
"""

from __future__ import annotations

import sys

import pandas as pd

from common import load_gold

from config import RAGConfig
from taxonomy import load_taxonomy


def main() -> int:
    config = RAGConfig()
    taxonomy = load_taxonomy(config.cluster_taxonomy_path)
    counts = (
        pd.read_csv(config.csv_knowledge_base_path, usecols=["cluster"], low_memory=False)["cluster"]
        .astype(str)
        .str.lower()
        .value_counts()
    )

    problems = []

    for question in load_gold():
        qid = question["id"]

        for cluster in question["expected_clusters"]:
            if not taxonomy.is_cluster(cluster):
                problems.append(f"{qid}: {cluster} is not in the taxonomy")

        for parent in question["expected_parents"]:
            if not taxonomy.is_parent(parent):
                problems.append(f"{qid}: {parent} is not a parent group")

        if question["type"] == "count":
            actual = str(int(counts.get(question["expected_clusters"][0], 0)))

            if question["expected_numbers"] != [actual]:
                problems.append(f"{qid}: expected {question['expected_numbers']}, data says {actual}")

        if question["type"] == "refusal" and not taxonomy.unknown_clusters(question["question"]):
            problems.append(f"{qid}: names no unknown cluster")

    for problem in problems:
        print(f"FAIL  {problem}")

    print(f"{len(load_gold())} gold questions checked, {len(problems)} problem(s)")
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
