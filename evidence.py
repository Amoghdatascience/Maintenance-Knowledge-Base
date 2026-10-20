"""
evidence.py

Evidence pool management: deduplicate, rank by relevance, select under budget.

The problem this fixes
----------------------
The old accumulator kept evidence in arrival order and stopped adding items
once MAX_EVIDENCE_CHARS (8000) was spent. One scoped vector search returns up
to 8 chunks, which can fill the whole budget by itself. So:

- a replan could never improve the answer: its better, targeted evidence
  arrived second and was dropped, while the first pass's weaker chunks stayed;
- in a comparison, the first cluster's chunks could crowd out the second's.

The new approach
----------------
1. Keep the full deduplicated pool across passes (state["evidence_pool"]).
2. Every pass, re-rank the WHOLE pool against the question.
3. Reserve one slot for each cluster / group the question names, so every side
   of a comparison is represented before anything else is added.
4. Give each named entity a fair share of the budget in a first fill, then let
   the best remaining items use whatever is left.
5. Hand the answer and judge prompts the selection, best first.

Relevance is a hybrid of BM25 (exact terms) and embedding cosine similarity
(meaning), fused by rank with RRF so the two never need comparable scales.
If embeddings are unavailable, BM25 alone is used; if both fail, arrival
order is kept. The pool never loses data, only the selection changes.

This module has no LangChain dependency so it can be unit-tested offline.
"""

from __future__ import annotations

import hashlib
import math
from typing import Callable, Dict, List, Optional, Sequence, Tuple

from bm25 import BM25, reciprocal_rank_fusion, tokenize
from graph_state import EvidenceItem


EmbedQuery = Callable[[str], List[float]]
EmbedDocuments = Callable[[List[str]], List[List[float]]]

# Structured questions are answered by exact CSV facts. Those items are short,
# authoritative, and must never lose to a narrative chunk that merely shares
# words with the question.
STRUCTURED_SIGNALS = (
    "how many", "count", "number of", "total", "list all", "show all",
    "record", "records", "work order", "workorder", "registration",
    "tail number", "ata", "schema", "columns",
)


def fingerprint(item: EvidenceItem) -> str:
    """
    Identity used for deduplication (unchanged from the old accumulator).
    """
    digest = hashlib.sha1(item["content"][:200].encode()).hexdigest()
    return f"{item['source']}|{digest}"


def merge_pool(
    pool: Sequence[EvidenceItem],
    new_items: Sequence[EvidenceItem],
) -> List[EvidenceItem]:
    """
    Append new items to the pool, skipping duplicates. Order is preserved.
    """
    merged = list(pool)
    seen = {fingerprint(item) for item in merged}

    for item in new_items:
        key = fingerprint(item)

        if key in seen:
            continue

        seen.add(key)
        merged.append(item)

    return merged


def select_arrival_order(
    items: Sequence[EvidenceItem],
    budget: int,
) -> List[EvidenceItem]:
    """
    The old behaviour, kept for ablation: first in wins until the budget runs out.
    """
    kept: List[EvidenceItem] = []

    for item in items:
        length = len(item["content"])

        if length > budget:
            continue

        kept.append(item)
        budget -= length

    return kept


def _cosine(a: Sequence[float], b: Sequence[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    norm = math.sqrt(sum(x * x for x in a)) * math.sqrt(sum(y * y for y in b))
    return dot / norm if norm else 0.0


class EvidenceRanker:
    """
    Ranks an evidence pool against a question and selects under a budget.

    Embeddings are cached by fingerprint, so re-ranking the pool on a replan
    only embeds the items that are new in that pass.
    """

    def __init__(
        self,
        embed_query: Optional[EmbedQuery] = None,
        embed_documents: Optional[EmbedDocuments] = None,
        rrf_k: int = 60,
    ):
        self.embed_query = embed_query
        self.embed_documents = embed_documents
        self.rrf_k = rrf_k
        self._doc_vectors: Dict[str, List[float]] = {}

    # ------------------------------------------------------------------
    # Ranking
    # ------------------------------------------------------------------

    def _embedding_order(
        self,
        query: str,
        items: Sequence[EvidenceItem],
    ) -> Optional[List[int]]:
        if self.embed_query is None or self.embed_documents is None:
            return None

        try:
            missing = [
                item for item in items if fingerprint(item) not in self._doc_vectors
            ]

            if missing:
                vectors = self.embed_documents([item["content"] for item in missing])

                for item, vector in zip(missing, vectors):
                    self._doc_vectors[fingerprint(item)] = list(vector)

            query_vector = self.embed_query(query)
        except Exception as error:  # noqa: BLE001 - fall back to BM25 only
            print(f"[evidence] embedding rank unavailable: {error}")
            return None

        similarities = [
            _cosine(query_vector, self._doc_vectors[fingerprint(item)]) for item in items
        ]

        return sorted(range(len(items)), key=lambda i: (-similarities[i], i))

    @staticmethod
    def _bm25_order(query: str, items: Sequence[EvidenceItem]) -> List[int]:
        index = BM25([tokenize(item["content"]) for item in items])
        query_tokens = tokenize(query)
        scores = [index.score(query_tokens, i) for i in range(len(items))]

        return sorted(range(len(items)), key=lambda i: (-scores[i], i))

    def rank(
        self,
        query: str,
        items: Sequence[EvidenceItem],
    ) -> List[Tuple[int, float]]:
        """
        Return (index into items, fused score), most relevant first.
        """
        if not items:
            return []

        orders = [self._bm25_order(query, items)]
        embedding_order = self._embedding_order(query, items)

        if embedding_order is not None:
            orders.append(embedding_order)

        fused = reciprocal_rank_fusion(
            [[str(i) for i in order] for order in orders],
            k=self.rrf_k,
        )

        return [(int(key), score) for key, score in fused]

    # ------------------------------------------------------------------
    # Selection
    # ------------------------------------------------------------------

    def select(
        self,
        question: str,
        items: Sequence[EvidenceItem],
        budget: int,
        asked_clusters: Sequence[str] = (),
        asked_parents: Sequence[str] = (),
        sub_questions: Sequence[str] = (),
    ) -> List[EvidenceItem]:
        """
        Choose the evidence the judge and answer prompts will see.

        Order of filling:
          0. pinned   - exact CSV facts, for structured questions
          1. coverage - the best item for each cluster/group the question names
          2. fair     - best-first, each named entity capped at an equal share
          3. rest     - best-first, no caps, until the budget is spent

        The returned list is ordered best first (pinned items at the top).
        """
        if not items:
            return []

        query = " ".join([question, *sub_questions]).strip()
        ranked = self.rank(query, items)
        position = {index: place for place, (index, _score) in enumerate(ranked)}
        order = [index for index, _score in ranked]

        chosen: List[int] = []
        remaining = budget

        def take(index: int) -> bool:
            nonlocal remaining
            length = len(items[index]["content"])

            if index in chosen or length > remaining:
                return False

            chosen.append(index)
            remaining -= length
            return True

        # 0. Pinned exact facts.
        lowered = question.lower()
        pinned: List[int] = []

        if any(signal in lowered for signal in STRUCTURED_SIGNALS):
            for index in order:
                if items[index]["tool"] == "csv_lookup" and take(index):
                    pinned.append(index)

        # Entities the question names. Clusters first: they are more specific.
        entities: List[Tuple[str, str]] = [("cluster", c) for c in asked_clusters]
        entities += [("parent", p) for p in asked_parents]

        def belongs(index: int, entity: Tuple[str, str]) -> bool:
            field, value = entity
            return bool(value) and items[index][field] == value

        # 1. Coverage: one slot per entity. Prefer substantive evidence over a
        #    bare taxonomy definition, but accept the definition if it is all
        #    there is (c_38 and c_50 have no documents at all).
        for entity in entities:
            if any(belongs(index, entity) for index in chosen):
                continue

            candidates = [index for index in order if belongs(index, entity)]
            substantive = [
                index for index in candidates if items[index]["tool"] != "metadata_search"
            ]

            for index in substantive + candidates:
                if take(index):
                    break

        # 2. Fair share across entities, so one side of a comparison cannot
        #    take the whole budget.
        if len(entities) >= 2:
            share = budget // len(entities)
            used: Dict[Tuple[str, str], int] = {
                entity: sum(
                    len(items[i]["content"]) for i in chosen if belongs(i, entity)
                )
                for entity in entities
            }

            for index in order:
                owners = [entity for entity in entities if belongs(index, entity)]

                if not owners:
                    continue

                length = len(items[index]["content"])

                if all(used[owner] + length <= share for owner in owners) and take(index):
                    for owner in owners:
                        used[owner] += length

        # 3. Everything else, best first.
        for index in order:
            take(index)

        # Pinned facts go first so the answer model reads them before prose;
        # everything else is ordered by relevance.
        others = sorted(
            (i for i in chosen if i not in pinned), key=lambda index: position[index]
        )

        return [items[i] for i in pinned + others]
