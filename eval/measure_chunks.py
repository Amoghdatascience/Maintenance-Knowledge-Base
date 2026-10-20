"""
eval/measure_chunks.py

Measures how well each chunking setting fits the embedding model's input
limit, using the model's own tokenizer. No index is built and no LLM is
needed, so it runs in seconds.

For each setting it:
1. chunks data/ exactly the way ingest.py does,
2. counts the tokens in every chunk with the embedding model's tokenizer
   (special tokens included, because they count against the limit),
3. reports how many chunks are longer than the model reads. Everything past
   the limit is silently dropped before embedding.

Usage (from the repo root):
    python eval/measure_chunks.py
    python eval/measure_chunks.py --settings markdown:600:100 recursive:1000:200
    python eval/measure_chunks.py --show-cut 2     # print text that gets dropped
"""

from __future__ import annotations

import argparse
import os
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
os.chdir(ROOT)

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config import RAGConfig  # noqa: E402
from ingest import DocumentIngestor, build_config  # noqa: E402

DEFAULT_SETTINGS = [
    "recursive:1000:200",  # the old setting: fixed windows
    "recursive:600:100",
    "markdown:400:50",
    "markdown:600:100",    # the new default: split by section
    "markdown:800:100",
]


def load_tokenizer(model_name: str):
    """
    The exact tokenizer and input limit the embedding model uses.
    """
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(model_name)
    return model.tokenizer, model.max_seq_length


def chunk_texts(spec: str, base: RAGConfig) -> list:
    """
    Chunk data/ with one setting ("strategy:size:overlap"), as ingest.py would.
    """
    strategy, size, overlap = spec.split(":")
    config = build_config(
        base.data_dir, base.db_dir, int(size), int(overlap), strategy, base.embedding_model
    )
    ingestor = DocumentIngestor(config)
    return [chunk.page_content for chunk in ingestor.split_documents(ingestor.load_documents())]


def main() -> None:
    parser = argparse.ArgumentParser(description="Measure chunk token lengths.")
    parser.add_argument("--settings", nargs="+", default=DEFAULT_SETTINGS)
    parser.add_argument(
        "--show-cut",
        type=int,
        default=0,
        help="for each setting, print the dropped text of this many over-limit chunks",
    )
    args = parser.parse_args()

    base = RAGConfig()
    tokenizer, limit = load_tokenizer(base.embedding_model)
    print(f"Model: {base.embedding_model}  |  input limit: {limit} tokens\n")

    rows = []

    for spec in args.settings:
        texts = chunk_texts(spec, base)
        token_ids = [tokenizer(text, add_special_tokens=True)["input_ids"] for text in texts]
        lengths = [len(ids) for ids in token_ids]
        over = [i for i, n in enumerate(lengths) if n > limit]

        rows.append(
            (
                spec,
                len(texts),
                round(statistics.mean(len(t) for t in texts)),
                round(statistics.median(lengths)),
                sorted(lengths)[int(0.95 * (len(lengths) - 1))],
                max(lengths),
                len(over) / len(lengths),
            )
        )

        # The model keeps [CLS] + the first (limit - 2) tokens + [SEP]. The
        # content tokens after that are the part that is never embedded.
        for i in over[: args.show_cut]:
            dropped = tokenizer.decode(token_ids[i][limit - 1 : -1])
            print(f"--- {spec}: chunk {i} has {lengths[i]} tokens; never embedded:")
            print(f"    ...{dropped}\n")

    header = ["setting", "chunks", "avg chars", "median tokens", "p95 tokens", "max tokens", "over limit"]
    print("| " + " | ".join(header) + " |")
    print("|" + "|".join("---" for _ in header) + "|")

    for spec, chunks, avg_chars, median, p95, longest, share in rows:
        print(f"| {spec} | {chunks} | {avg_chars} | {median} | {p95} | {longest} | {share:.0%} |")


if __name__ == "__main__":
    main()
