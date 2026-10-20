"""
bm25.py

Pure-Python BM25 (Okapi) plus Reciprocal Rank Fusion, with a tokenizer tuned
for aviation maintenance text. No extra dependency.

Why BM25 next to embeddings:
- MiniLM is a general-purpose model. It is weak on exact shorthand such as
  "mag drop", "cht", "r&r", ATA codes like 7230, or tail numbers like N556ND.
- BM25 scores exact token overlap, so it is strong exactly where embeddings
  are weak, and weak where embeddings are strong (paraphrases, synonyms).
- Fusing the two rankings with RRF gets most of the benefit of both without
  tuning score scales, because RRF only looks at rank positions.
"""

from __future__ import annotations

import math
import re
from collections import Counter
from typing import Dict, Iterable, List, Optional, Sequence, Tuple


# Keeps compound maintenance tokens whole: a/c, l/h, r&r, c_22, 7230.0, n556nd
TOKEN_PATTERN = re.compile(r"[a-z0-9]+(?:[/&_.][a-z0-9]+)*")

# Map common maintenance shorthand onto one canonical token so that "mag" in a
# question matches "magneto" in a document and vice versa. Applied to both
# queries and documents, so the mapping only has to be consistent, not pretty.
ABBREVIATIONS: Dict[str, str] = {
    "a/c": "aircraft",
    "acft": "aircraft",
    "eng": "engine",
    "cyl": "cylinder",
    "cyls": "cylinder",
    "mag": "magneto",
    "mags": "magneto",
    "carb": "carburetor",
    "l/h": "left",
    "lh": "left",
    "r/h": "right",
    "rh": "right",
    "fwd": "forward",
    "insp": "inspection",
    "ops": "operation",
    "r&r": "replace",
    "replaced": "replace",
    "chk": "check",
    "ck": "check",
    "checked": "check",
    "prop": "propeller",
    "props": "propeller",
    "batt": "battery",
    "cowl": "cowling",
    "temp": "temperature",
    "pres": "pressure",
    "press": "pressure",
    "plugs": "plug",
    "leaks": "leak",
    "leaking": "leak",
    "cracked": "crack",
    "cracks": "crack",
}

STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "by", "for", "from", "has",
    "in", "is", "it", "of", "on", "or", "that", "the", "this", "to", "was",
    "were", "with", "what", "which", "how", "does", "do", "me", "about",
    "tell", "show", "give", "there", "their", "its", "into", "than", "then",
    "can", "you", "i", "my", "any", "all", "these", "those", "should",
}


def normalize_token(token: str) -> str:
    """
    Canonicalize one token: ATA codes lose their pandas '.0', shorthand is
    expanded, and cluster IDs like c22 become c_22.
    """
    if re.fullmatch(r"\d+\.0", token):
        token = token[:-2]

    if re.fullmatch(r"c\d+", token):
        token = f"c_{token[1:]}"

    return ABBREVIATIONS.get(token, token)


def tokenize(text: str) -> List[str]:
    """
    Lowercase, extract tokens, normalize, and drop stopwords.
    """
    tokens: List[str] = []

    for raw in TOKEN_PATTERN.findall(str(text).lower()):
        token = normalize_token(raw.strip("._/&"))

        if not token or token in STOPWORDS:
            continue

        tokens.append(token)

    return tokens


class BM25:
    """
    Okapi BM25 over a fixed corpus of pre-tokenized documents.

    Uses the Lucene IDF variant, log(1 + (N - df + 0.5) / (df + 0.5)), which
    is always positive. The classic Okapi IDF goes negative for terms in more
    than half the documents, which on a small corpus (348 chunks, where words
    like "baffle" appear everywhere in one group) makes common-but-relevant
    words actively penalize a document.
    """

    def __init__(
        self,
        corpus_tokens: Sequence[Sequence[str]],
        k1: float = 1.5,
        b: float = 0.75,
    ):
        self.k1 = k1
        self.b = b
        self.doc_count = len(corpus_tokens)
        self.term_freqs: List[Counter] = [Counter(doc) for doc in corpus_tokens]
        self.doc_lengths: List[int] = [len(doc) for doc in corpus_tokens]
        self.avg_doc_length = (
            sum(self.doc_lengths) / self.doc_count if self.doc_count else 0.0
        )

        doc_freq: Counter = Counter()

        for freqs in self.term_freqs:
            doc_freq.update(freqs.keys())

        self.idf: Dict[str, float] = {
            term: math.log(1.0 + (self.doc_count - df + 0.5) / (df + 0.5))
            for term, df in doc_freq.items()
        }

    def score(self, query_tokens: Iterable[str], index: int) -> float:
        """
        BM25 score of one document for a query.

        Each distinct query term counts once. Query expansion appends taxonomy
        descriptions that repeat words, and counting repeats would let the
        expansion drown out the user's own words.
        """
        if not self.doc_count or self.avg_doc_length == 0:
            return 0.0

        freqs = self.term_freqs[index]
        length_norm = 1 - self.b + self.b * self.doc_lengths[index] / self.avg_doc_length
        total = 0.0

        for term in set(query_tokens):
            tf = freqs.get(term, 0)

            if not tf:
                continue

            idf = self.idf.get(term, 0.0)
            total += idf * tf * (self.k1 + 1) / (tf + self.k1 * length_norm)

        return total

    def top_n(
        self,
        query: str,
        n: int,
        allowed: Optional[Iterable[int]] = None,
    ) -> List[Tuple[int, float]]:
        """
        Best-scoring document indices for a query, highest first.

        allowed restricts scoring to a subset (the metadata scope), which is
        how BM25 honours the same cluster/group filter as the vector search.
        Documents with a zero score are never returned.
        """
        query_tokens = tokenize(query)

        if not query_tokens:
            return []

        candidates = range(self.doc_count) if allowed is None else allowed
        scored = [
            (index, self.score(query_tokens, index)) for index in candidates
        ]
        scored = [pair for pair in scored if pair[1] > 0]
        scored.sort(key=lambda pair: (-pair[1], pair[0]))

        return scored[:n]


def reciprocal_rank_fusion(
    rankings: Sequence[Sequence[str]],
    k: int = 60,
    weights: Optional[Sequence[float]] = None,
) -> List[Tuple[str, float]]:
    """
    Fuse several ranked lists of keys into one.

    score(key) = sum over lists of weight / (k + rank), rank starting at 1.

    k=60 is the value from the original RRF paper (Cormack et al., 2009). It
    damps the gap between rank 1 and rank 2, so one retriever being very
    confident cannot override the other being mildly confident. Ties keep the
    order of first appearance, which makes results reproducible.
    """
    weights = list(weights) if weights is not None else [1.0] * len(rankings)
    scores: Dict[str, float] = {}
    first_seen: Dict[str, int] = {}
    position = 0

    for ranking, weight in zip(rankings, weights):
        for rank, key in enumerate(ranking, start=1):
            scores[key] = scores.get(key, 0.0) + weight / (k + rank)

            if key not in first_seen:
                first_seen[key] = position
                position += 1

    return sorted(scores.items(), key=lambda pair: (-pair[1], first_seen[pair[0]]))
