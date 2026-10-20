"""
eval/eval_chunking.py

Chunking sweep. For each chunking configuration it builds a separate Chroma
index (under eval/.chunk_dbs/, never touching chroma_db/), then reports:

- chunk count and size
- real token counts from the embedding model's own tokenizer, and the share
  of chunks longer than its input limit (256 word pieces for MiniLM). Those
  chunks are silently truncated: their tail is never embedded.
- the Part B retrieval metrics from eval_retrieval.py with hybrid + ranked

Usage (from the repo root):
    python eval/eval_chunking.py
    python eval/eval_chunking.py --configs markdown:600:100 recursive:1000:200
"""

from __future__ import annotations

import argparse
import shutil
import statistics
from typing import List, Tuple

from common import (
    ROOT,
    build_stack,
    fmt,
    load_gold,
    markdown_table,
    run_retrieval,
    save_results,
    score_evidence,
    summarize,
)

from config import RAGConfig
from ingest import DocumentIngestor, build_config

DEFAULT_CONFIGS = [
    "recursive:1000:200",  # the old setting
    "recursive:600:100",
    "markdown:400:50",
    "markdown:600:100",    # the new default
    "markdown:800:100",
]

DB_ROOT = ROOT / "eval" / ".chunk_dbs"


def parse(spec: str) -> Tuple[str, int, int]:
    strategy, size, overlap = spec.split(":")
    return strategy, int(size), int(overlap)


def token_stats(retriever, texts: List[str]) -> dict:
    """
    Token lengths using the embedding model's real tokenizer.
    """
    client = getattr(retriever.embeddings, "_client", None) or getattr(
        retriever.embeddings, "client", None
    )
    tokenizer = getattr(client, "tokenizer", None)
    limit = getattr(client, "max_seq_length", 256) or 256

    if tokenizer is None:
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(retriever.embedding_model)

    lengths = [len(tokenizer(text, add_special_tokens=True)["input_ids"]) for text in texts]

    return {
        "limit": limit,
        "median_tokens": statistics.median(lengths),
        "p95_tokens": sorted(lengths)[int(0.95 * (len(lengths) - 1))],
        "max_tokens": max(lengths),
        "over_limit": sum(1 for n in lengths if n > limit) / len(lengths),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    parser.add_argument("--configs", nargs="+", default=DEFAULT_CONFIGS)
    parser.add_argument("--keep", action="store_true", help="keep the sweep indexes")
    args = parser.parse_args()

    base = RAGConfig()
    gold = [q for q in load_gold() if q["type"] != "refusal"]
    results = []

    for spec in args.configs:
        strategy, size, overlap = parse(spec)
        db_dir = DB_ROOT / spec.replace(":", "_")

        if db_dir.exists():
            shutil.rmtree(db_dir)

        config = build_config(
            base.data_dir, str(db_dir), size, overlap, strategy, base.embedding_model
        )
        print(f"\n[sweep] {spec}: ingesting into {db_dir}")
        DocumentIngestor(config).run()

        stack = build_stack(config)
        chunks = stack.retriever.vectorstore.get(include=["documents"])["documents"]
        tokens = token_stats(stack.retriever, chunks)

        rows = []

        for question in gold:
            outcome = run_retrieval(stack, question["question"], "hybrid", "ranked")
            rows.append({"metrics": score_evidence(question, outcome["evidence"], stack.taxonomy)})

        scores = summarize(rows, ["coverage", "mrr", "scope_precision", "fact_recall"])

        results.append(
            {
                "config": spec,
                "chunks": len(chunks),
                "avg_chars": statistics.mean(len(c) for c in chunks),
                **tokens,
                **scores,
            }
        )

        if not args.keep:
            shutil.rmtree(db_dir, ignore_errors=True)

    table = markdown_table(
        ["config", "chunks", "avg chars", "median tok", "p95 tok", "max tok", "over limit",
         "coverage", "MRR", "scope precision", "fact recall"],
        [
            [
                r["config"], str(r["chunks"]), fmt(r["avg_chars"], 0), fmt(r["median_tokens"], 0),
                fmt(r["p95_tokens"], 0), fmt(r["max_tokens"], 0), f"{r['over_limit']:.0%}",
                fmt(r["coverage"]), fmt(r["mrr"]), fmt(r["scope_precision"]), fmt(r["fact_recall"]),
            ]
            for r in results
        ],
    )

    # Best = highest coverage + MRR among configs that never truncate; ties go
    # to fewer chunks (less fragmentation, cheaper index).
    safe = [r for r in results if r["over_limit"] == 0] or results
    best = max(safe, key=lambda r: ((r["coverage"] or 0) + (r["mrr"] or 0), -r["chunks"]))

    report = "\n\n".join(
        [
            "# Chunking sweep (hybrid retrieval + ranked selection)",
            f"Token limit of {base.embedding_model}: {results[0]['limit'] if results else '?'}",
            table,
            f"Recommended: **{best['config']}** (best coverage + MRR among configs with no "
            "truncated chunks). Set CHUNK_STRATEGY / CHUNK_SIZE / CHUNK_OVERLAP in .env to match, "
            "then `python ingest.py --reset`.",
        ]
    )

    print()
    print(report)

    if DB_ROOT.exists() and not args.keep and not any(DB_ROOT.iterdir()):
        DB_ROOT.rmdir()

    path = save_results("chunking", {"results": results}, report)
    print(f"\nSaved {path}")


if __name__ == "__main__":
    main()
