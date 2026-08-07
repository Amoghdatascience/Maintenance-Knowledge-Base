"""
taxonomy.py

Single source of truth for cluster IDs, parent groups, and descriptions.

Everything here is derived from data/cluster_taxonomy.csv at load time. This
replaces the hardcoded c_0..c_50 description map that used to live in
nodes.expand_cluster_ids, so cluster definitions exist in exactly one place.

Used by:
- ingest.py     -> join a filename to its cluster/parent metadata
- planner.py    -> validate entities the LLM names, build the deterministic plan
- tools.py      -> expand retrieval queries, scope metadata searches
"""

from __future__ import annotations

import re
from functools import lru_cache
from pathlib import Path
from typing import Dict, List, Optional, Set

import pandas as pd


# Words too generic to identify a parent group on their own. Without this,
# "system" would match both oil_system and start_system.
GENERIC_WORDS: Set[str] = {
    "system",
    "general",
    "control",
    "reported",
    "cleaning",
    "unspecified",
    "issue",
    "issues",
    "problem",
    "problems",
    "damage",
    "loose",
    "missing",
    "clean",
    "remove",
    "replace",
    "repair",
    "check",
    "inspect",
    "unknown",
    "appearance",
}


class Taxonomy:
    """
    Read-only view over cluster_taxonomy.csv.
    """

    def __init__(self, taxonomy_path: str):
        self.path = Path(taxonomy_path)

        if not self.path.exists():
            raise FileNotFoundError(f"Cluster taxonomy not found: {self.path}")

        frame = pd.read_csv(self.path)

        self._parent_of: Dict[str, str] = {}
        self._description_of: Dict[str, str] = {}
        self._clusters_in: Dict[str, List[str]] = {}

        for row in frame.itertuples(index=False):
            cluster = str(row.cluster).strip()
            parent = str(row.parent).strip()
            description = str(row.description).strip()

            self._parent_of[cluster] = parent
            self._description_of[cluster] = description
            self._clusters_in.setdefault(parent, []).append(cluster)

        self._discriminators = self._build_discriminators()

    # ------------------------------------------------------------------
    # Basic accessors
    # ------------------------------------------------------------------

    @property
    def clusters(self) -> List[str]:
        return sorted(self._parent_of, key=_cluster_sort_key)

    @property
    def parents(self) -> List[str]:
        return sorted(self._clusters_in)

    def is_cluster(self, cluster_id: str) -> bool:
        return cluster_id in self._parent_of

    def is_parent(self, parent: str) -> bool:
        return parent in self._clusters_in

    def parent_of(self, cluster_id: str) -> str:
        return self._parent_of.get(cluster_id, "")

    def description_of(self, cluster_id: str) -> str:
        return self._description_of.get(cluster_id, "")

    def clusters_in(self, parent: str) -> List[str]:
        return sorted(self._clusters_in.get(parent, []), key=_cluster_sort_key)

    # ------------------------------------------------------------------
    # Normalization and validation
    # ------------------------------------------------------------------

    @staticmethod
    def normalize_cluster(cluster_id: str) -> str:
        """
        Normalize 22 / c22 / c_22 / cluster 22 into c_22.

        Mirrors CSVKnowledgeBase._normalize_cluster_id so both layers agree.
        """
        numbers = re.findall(r"\d+", str(cluster_id).lower().strip())

        if not numbers:
            return str(cluster_id).lower().strip()

        return f"c_{numbers[0]}"

    def valid_clusters(self, cluster_ids: List[str]) -> List[str]:
        """
        Normalize then drop any cluster ID that does not exist.

        This is what stops a hallucinated c_99 from reaching the tools.
        """
        kept: List[str] = []

        for cluster_id in cluster_ids:
            normalized = self.normalize_cluster(cluster_id)

            if self.is_cluster(normalized) and normalized not in kept:
                kept.append(normalized)

        return kept

    def valid_parents(self, parent_groups: List[str]) -> List[str]:
        """
        Resolve loosely-written parent groups ("flight control", "Baffle")
        against the real parent names, dropping anything unrecognized.
        """
        kept: List[str] = []

        for candidate in parent_groups:
            resolved = self.resolve_parent(candidate)

            if resolved and resolved not in kept:
                kept.append(resolved)

        return kept

    def resolve_parent(self, candidate: str) -> Optional[str]:
        """
        Map a free-form group name onto a canonical parent, or None.
        """
        cleaned = str(candidate).strip().lower().replace(" ", "_").replace("-", "_")

        if not cleaned:
            return None

        for parent in self._clusters_in:
            if parent.lower() == cleaned:
                return parent

        return None

    # ------------------------------------------------------------------
    # Free-text matching
    # ------------------------------------------------------------------

    def _build_discriminators(self) -> Dict[str, str]:
        """
        Index description words that unambiguously identify one parent group.

        A word shared by several parents (like "engine") is discarded, so the
        index self-tunes as the taxonomy changes.
        """
        word_to_parents: Dict[str, Set[str]] = {}

        for cluster, description in self._description_of.items():
            parent = self._parent_of[cluster]

            for word in re.findall(r"[a-z]+", description.lower()):
                if len(word) < 4 or word in GENERIC_WORDS:
                    continue

                word_to_parents.setdefault(word, set()).add(parent)

        return {
            word: next(iter(parents))
            for word, parents in word_to_parents.items()
            if len(parents) == 1
        }

    def match_parents(self, text: str) -> List[str]:
        """
        Find parent groups mentioned in free text.

        Matches on the parent name itself ("flight control", "oil_system"), on
        its non-generic name tokens ("baffle", "propeller"), and on description
        words that belong to only one group ("magneto" -> ignition).
        """
        lowered = str(text).lower()
        matched: List[str] = []

        def remember(parent: str) -> None:
            if parent not in matched:
                matched.append(parent)

        for parent in self._clusters_in:
            spaced = parent.lower().replace("_", " ")

            if spaced in lowered or parent.lower() in lowered:
                remember(parent)
                continue

            for token in parent.lower().split("_"):
                if len(token) < 4 or token in GENERIC_WORDS:
                    continue

                if re.search(rf"\b{re.escape(token)}\b", lowered):
                    remember(parent)
                    break

        for word, parent in self._discriminators.items():
            if re.search(rf"\b{re.escape(word)}\b", lowered):
                remember(parent)

        return matched

    @staticmethod
    def mentioned_clusters(text: str) -> List[str]:
        """
        Extract every cluster ID written in the text, real or not.

        Supports c_22, c22, cluster 22, cluster_22, cluster c_22, cluster_c_22.

        Unlike match_clusters this does NOT filter against the taxonomy, so a
        caller can tell "no cluster was asked about" apart from "a cluster that
        does not exist was asked about" -- two cases that need different
        answers.
        """
        lowered = str(text).lower()
        found: List[str] = []

        for match in re.findall(r"\bc_?(\d+)\b", lowered):
            if f"c_{match}" not in found:
                found.append(f"c_{match}")

        for match in re.findall(r"\bcluster[\s_]+c?_?(\d+)\b", lowered):
            if f"c_{match}" not in found:
                found.append(f"c_{match}")

        return found

    def unknown_clusters(self, text: str) -> List[str]:
        """
        Cluster IDs the text names that do not exist in the taxonomy.
        """
        return [
            cluster_id
            for cluster_id in self.mentioned_clusters(text)
            if not self.is_cluster(cluster_id)
        ]

    def match_clusters(self, text: str) -> List[str]:
        """
        Extract cluster IDs from text, keeping only ones that really exist.
        """
        return self.valid_clusters(self.mentioned_clusters(text))

    # ------------------------------------------------------------------
    # Query expansion
    # ------------------------------------------------------------------

    def expansion_terms(
        self,
        cluster_ids: Optional[List[str]] = None,
        parent_groups: Optional[List[str]] = None,
    ) -> str:
        """
        Build retrieval keywords for the named clusters and parent groups.

        Replaces the hand-written per-group keyword blocks that used to live in
        nodes.expand_query -- the terms now come from the taxonomy itself.
        """
        terms: List[str] = []

        for cluster_id in cluster_ids or []:
            if not self.is_cluster(cluster_id):
                continue

            terms.append(f"{cluster_id} {self.description_of(cluster_id)}")

        for parent in parent_groups or []:
            if not self.is_parent(parent):
                continue

            members = self.clusters_in(parent)
            descriptions = " ".join(self.description_of(c) for c in members)

            terms.append(
                f"{parent} parent group {' '.join(members)} {descriptions}"
            )

        return " ".join(terms).strip()


def _cluster_sort_key(cluster_id: str):
    """
    Sort c_2 before c_10 rather than lexicographically.
    """
    numbers = re.findall(r"\d+", cluster_id)
    return (int(numbers[0]) if numbers else 0, cluster_id)


@lru_cache(maxsize=4)
def load_taxonomy(taxonomy_path: str) -> Taxonomy:
    """
    Load and cache the taxonomy for a given path.
    """
    return Taxonomy(taxonomy_path)
