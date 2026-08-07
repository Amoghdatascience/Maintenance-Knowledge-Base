"""
tools.py

The three retrieval tools a plan step can invoke.

Each dispatcher returns a list of EvidenceItem carrying provenance, so the
accumulator can deduplicate and the answer can name its sources.

- csv_lookup      deterministic structured facts (CSVKnowledgeBase)
- vector_search   similarity search, optionally scoped by cluster metadata
- metadata_search taxonomy definitions and whole-group documents
"""

from __future__ import annotations

from typing import List, Sequence

from config import RAGConfig
from csv_lookup import CSVKnowledgeBase
from graph_state import EvidenceItem, PlanStep
from retriever import VectorRetriever
from taxonomy import Taxonomy


# CSVKnowledgeBase.lookup returns this when its dispatcher finds no match.
CSV_MISS_PREFIX = "CSV lookup could not confidently answer"

MAX_ITEM_CHARS = 2000
MAX_CLUSTERS_PER_STEP = 4

COUNT_SIGNALS = ("how many", "count", "number of", "total")
RECORD_SIGNALS = ("record", "records", "show", "example", "examples", "raw")


def _evidence(
    tool: str,
    sub_question: str,
    content: str,
    source: str,
    cluster: str = "",
    parent: str = "",
) -> EvidenceItem:
    return {
        "tool": tool,
        "sub_question": sub_question,
        "content": content.strip()[:MAX_ITEM_CHARS],
        "source": source,
        "cluster": cluster,
        "parent": parent,
    }


class ToolExecutor:
    """
    Executes validated plan steps against the knowledge base.
    """

    def __init__(
        self,
        config: RAGConfig,
        taxonomy: Taxonomy,
        knowledge_base: CSVKnowledgeBase,
        retriever: VectorRetriever,
    ):
        self.config = config
        self.taxonomy = taxonomy
        self.knowledge_base = knowledge_base
        self.retriever = retriever

    def execute(
        self,
        step: PlanStep,
        research_notes: Sequence[str] = (),
    ) -> List[EvidenceItem]:
        """
        Run one plan step. Tool failures degrade to no evidence, not a crash.
        """
        try:
            if step["tool"] == "csv_lookup":
                return self.csv_lookup(step)

            if step["tool"] == "vector_search":
                return self.vector_search(step, research_notes)

            if step["tool"] == "metadata_search":
                return self.metadata_search(step)
        except Exception as error:  # noqa: BLE001 - one bad step must not end the run
            print(f"[tools] {step['tool']} failed: {error}")
            return []

        print(f"[tools] unknown tool: {step['tool']}")
        return []

    # ------------------------------------------------------------------
    # csv_lookup
    # ------------------------------------------------------------------

    def csv_lookup(self, step: PlanStep) -> List[EvidenceItem]:
        """
        Deterministic structured lookup.

        With explicit cluster IDs, call the precise method instead of routing
        through the keyword dispatcher, which can only handle one ID at a time.
        """
        sub_question = step["sub_question"]
        lowered = sub_question.lower()
        source = self.config.csv_knowledge_base_path
        items: List[EvidenceItem] = []

        clusters = step["cluster_ids"][:MAX_CLUSTERS_PER_STEP]

        if clusters:
            for cluster_id in clusters:
                if any(signal in lowered for signal in COUNT_SIGNALS):
                    content = self.knowledge_base.count_records_for_cluster(cluster_id)
                elif any(signal in lowered for signal in RECORD_SIGNALS):
                    content = self.knowledge_base.show_records_for_cluster(cluster_id)
                else:
                    content = self.knowledge_base.describe_cluster(cluster_id)

                if content and not content.startswith(CSV_MISS_PREFIX):
                    items.append(
                        _evidence(
                            "csv_lookup",
                            sub_question,
                            content,
                            source,
                            cluster=cluster_id,
                            parent=self.taxonomy.parent_of(cluster_id),
                        )
                    )

            return items

        content = self.knowledge_base.lookup(sub_question)

        if content and not content.startswith(CSV_MISS_PREFIX):
            items.append(_evidence("csv_lookup", sub_question, content, source))

        return items

    # ------------------------------------------------------------------
    # vector_search
    # ------------------------------------------------------------------

    def build_query(
        self,
        step: PlanStep,
        research_notes: Sequence[str] = (),
    ) -> str:
        """
        Build the retrieval string for a step.

        Expansion terms come from the taxonomy rather than a hardcoded map, so
        cluster descriptions live in exactly one place.
        """
        parts = [step["sub_question"]]

        expansion = self.taxonomy.expansion_terms(
            cluster_ids=step["cluster_ids"],
            parent_groups=step["parent_groups"],
        )

        if expansion:
            parts.append(expansion)

        if step["keywords"]:
            parts.append(" ".join(step["keywords"]))

        if research_notes:
            parts.append(" ".join(research_notes))

        return " ".join(parts).strip()

    def vector_search(
        self,
        step: PlanStep,
        research_notes: Sequence[str] = (),
    ) -> List[EvidenceItem]:
        """
        Similarity search, scoped by cluster metadata when the step names any.

        The metadata filter is what makes retrieving an exact cluster ID safe:
        a search scoped to c_38 physically cannot return c_37's document.
        """
        query = self.build_query(step, research_notes)

        documents = self.retriever.retrieve(
            query,
            cluster_ids=step["cluster_ids"],
            parent_groups=step["parent_groups"],
        )

        items: List[EvidenceItem] = []

        for document in documents:
            metadata = document.metadata or {}

            items.append(
                _evidence(
                    "vector_search",
                    step["sub_question"],
                    document.page_content,
                    str(metadata.get("source", "unknown source")),
                    cluster=str(metadata.get("cluster", "")),
                    parent=str(metadata.get("parent", "")),
                )
            )

        return items

    # ------------------------------------------------------------------
    # metadata_search
    # ------------------------------------------------------------------

    def metadata_search(self, step: PlanStep) -> List[EvidenceItem]:
        """
        Taxonomy definitions plus whole-group documents.

        Answers "which clusters belong to this group" before anything tries to
        search inside the group.
        """
        sub_question = step["sub_question"]
        items: List[EvidenceItem] = []

        for cluster_id in step["cluster_ids"][:MAX_CLUSTERS_PER_STEP]:
            content = self.knowledge_base.describe_cluster_from_taxonomy(cluster_id)

            if content:
                items.append(
                    _evidence(
                        "metadata_search",
                        sub_question,
                        content,
                        self.config.cluster_taxonomy_path,
                        cluster=cluster_id,
                        parent=self.taxonomy.parent_of(cluster_id),
                    )
                )

        for parent in step["parent_groups"]:
            members = self.taxonomy.clusters_in(parent)

            listing = "\n".join(
                f"{cluster_id}: {self.taxonomy.description_of(cluster_id)}"
                for cluster_id in members
            )

            items.append(
                _evidence(
                    "metadata_search",
                    sub_question,
                    f"Parent group '{parent}' contains {len(members)} clusters:\n{listing}",
                    self.config.cluster_taxonomy_path,
                    parent=parent,
                )
            )

            for document in self.retriever.fetch_by_metadata(
                parent_groups=[parent],
                doc_type="parent_group",
                limit=3,
            ):
                metadata = document.metadata or {}

                items.append(
                    _evidence(
                        "metadata_search",
                        sub_question,
                        document.page_content,
                        str(metadata.get("source", "unknown source")),
                        parent=parent,
                    )
                )

        if not items and not step["cluster_ids"] and not step["parent_groups"]:
            # Nothing named: fall back to the full cluster listing.
            items.append(
                _evidence(
                    "metadata_search",
                    sub_question,
                    self.knowledge_base.list_clusters_from_taxonomy(),
                    self.config.cluster_taxonomy_path,
                )
            )

        return items
