"""
csv_lookup.py

CSV knowledge-base lookup for structured Agentic RAG questions.

This module uses:
1. final_joined_output.csv as the main structured maintenance-record knowledge base.
2. cluster_taxonomy.csv as the authoritative cluster mapping.

Use this file for:
- exact cluster lookup
- cluster parent/description lookup
- cluster lists
- record counts
- raw record lookup
- work order lookup
- registration lookup
- ATA code lookup
- keyword search over PROBLEM/ACTION
- CSV schema/column inspection
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Optional

import pandas as pd

from config import RAGConfig


class CSVKnowledgeBase:
    """
    Structured CSV knowledge base for maintenance records.
    """

    def __init__(self, config: RAGConfig):
        self.config = config

        self.csv_path = Path(config.csv_knowledge_base_path)

        if not self.csv_path.exists():
            raise FileNotFoundError(
                f"CSV knowledge base not found: {self.csv_path}"
            )

        self.df = pd.read_csv(self.csv_path, low_memory=False)
        self.df = self._normalize_columns(self.df)

        taxonomy_path_value = getattr(
            config,
            "cluster_taxonomy_path",
            "data/cluster_taxonomy.csv",
        )

        self.taxonomy_path = Path(taxonomy_path_value)
        self.taxonomy_df = None

        if self.taxonomy_path.exists():
            self.taxonomy_df = pd.read_csv(self.taxonomy_path, low_memory=False)
            self.taxonomy_df = self._normalize_columns(self.taxonomy_df)

    @staticmethod
    def _normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
        """
        Normalize column names so lookup works even if columns vary slightly.
        """
        column_map = {}

        for col in df.columns:
            normalized = (
                str(col)
                .strip()
                .lower()
                .replace("#", "")
                .replace(" ", "_")
                .replace("-", "_")
            )

            if normalized in {
                "wko",
                "wko_",
                "work_order",
                "workorder",
                "work_order_number",
            }:
                column_map[col] = "workorder"

            elif normalized in {"ata_code", "ata"}:
                column_map[col] = "ata_code"

            elif normalized in {
                "problem",
                "problem_description",
                "problem_desc",
            }:
                column_map[col] = "problem"

            elif normalized in {
                "action",
                "corrective_action",
                "maintenance_action",
            }:
                column_map[col] = "action"

            elif normalized in {"cluster", "cluster_id"}:
                column_map[col] = "cluster"

            elif normalized in {
                "cluster_name",
                "cluster_desc",
                "cluster_description",
            }:
                column_map[col] = "cluster_name"

            elif normalized in {
                "parent",
                "parent_group",
                "group",
                "category",
            }:
                column_map[col] = "parent"

            elif normalized in {
                "description",
                "desc",
                "cluster_definition",
                "definition",
            }:
                column_map[col] = "description"

            elif normalized in {
                "registration",
                "registration_number",
                "registration_",
                "tail",
                "tail_number",
            }:
                column_map[col] = "registration"

            elif normalized in {"date", "maintenance_date"}:
                column_map[col] = "date"

            elif normalized in {
                "date_time_opened",
                "date_opened",
                "opened",
                "open_date",
            }:
                column_map[col] = "date_time_opened"

            elif normalized in {
                "date_time_closed",
                "date_closed",
                "closed",
                "close_date",
            }:
                column_map[col] = "date_time_closed"

            else:
                column_map[col] = normalized

        return df.rename(columns=column_map)

    def schema_summary(self) -> str:
        """
        Return a summary of available CSV columns and row counts.
        """
        lines = [
            "CSV knowledge base loaded successfully.",
            f"Main CSV path: {self.csv_path}",
            f"Main CSV rows: {len(self.df)}",
            f"Main CSV columns: {', '.join(self.df.columns)}",
        ]

        if self.taxonomy_df is not None:
            lines.extend(
                [
                    "",
                    "Cluster taxonomy loaded successfully.",
                    f"Taxonomy path: {self.taxonomy_path}",
                    f"Taxonomy rows: {len(self.taxonomy_df)}",
                    f"Taxonomy columns: {', '.join(self.taxonomy_df.columns)}",
                ]
            )
        else:
            lines.extend(
                [
                    "",
                    "Cluster taxonomy was not loaded.",
                    f"Expected taxonomy path: {self.taxonomy_path}",
                ]
            )

        return "\n".join(lines)

    def count_rows(self) -> str:
        return f"Total records in final_joined_output.csv: {len(self.df)}"

    def count_clusters(self) -> str:
        """
        Count unique clusters. Prefer taxonomy if available.
        """
        if self.taxonomy_df is not None and "cluster" in self.taxonomy_df.columns:
            count = self.taxonomy_df["cluster"].dropna().astype(str).nunique()
            return f"Total unique clusters in cluster_taxonomy.csv: {count}"

        if "cluster" not in self.df.columns:
            return "The CSV does not contain a cluster column."

        count = self.df["cluster"].dropna().astype(str).nunique()
        return f"Total unique clusters in final_joined_output.csv: {count}"

    def list_clusters(self) -> str:
        """
        List all clusters. Prefer taxonomy if available.
        """
        if self.taxonomy_df is not None and "cluster" in self.taxonomy_df.columns:
            return self.list_clusters_from_taxonomy()

        if "cluster" not in self.df.columns:
            return "The CSV does not contain a cluster column."

        if "cluster_name" in self.df.columns:
            cluster_df = (
                self.df[["cluster", "cluster_name"]]
                .dropna(subset=["cluster"])
                .drop_duplicates()
                .sort_values("cluster")
            )

            lines = [
                f"Total unique clusters: {cluster_df['cluster'].nunique()}",
                "",
                "Clusters:",
            ]

            for _, row in cluster_df.iterrows():
                lines.append(f"- {row['cluster']}: {row['cluster_name']}")

            return "\n".join(lines)

        clusters = sorted(self.df["cluster"].dropna().astype(str).unique())

        lines = [
            f"Total unique clusters: {len(clusters)}",
            "",
            "Clusters:",
        ]

        for cluster in clusters:
            lines.append(f"- {cluster}")

        return "\n".join(lines)

    def list_clusters_from_taxonomy(self) -> str:
        """
        List clusters from cluster_taxonomy.csv.
        """
        if self.taxonomy_df is None:
            return "cluster_taxonomy.csv is not available."

        required_cols = {"cluster", "parent", "description"}

        if not required_cols.issubset(self.taxonomy_df.columns):
            return (
                "cluster_taxonomy.csv must contain cluster, parent, "
                "and description columns."
            )

        taxonomy_df = (
            self.taxonomy_df[["cluster", "parent", "description"]]
            .dropna(subset=["cluster"])
            .drop_duplicates()
            .sort_values("cluster")
        )

        lines = [
            f"Total defined clusters: {taxonomy_df['cluster'].nunique()}",
            "",
            "Clusters:",
        ]

        for _, row in taxonomy_df.iterrows():
            lines.append(
                f"- {row['cluster']}: parent={row['parent']}, "
                f"description={row['description']}"
            )

        return "\n".join(lines)

    def describe_cluster_from_taxonomy(self, cluster_id: str) -> str:
        """
        Describe a cluster using cluster_taxonomy.csv.
        """
        if self.taxonomy_df is None:
            return ""

        required_cols = {"cluster", "parent", "description"}

        if not required_cols.issubset(self.taxonomy_df.columns):
            return ""

        normalized_cluster_id = self._normalize_cluster_id(cluster_id)

        filtered = self.taxonomy_df[
            self.taxonomy_df["cluster"].astype(str).str.lower()
            == normalized_cluster_id.lower()
        ]

        if filtered.empty:
            return ""

        row = filtered.iloc[0]

        return (
            f"Cluster: {row['cluster']}\n"
            f"Parent group: {row['parent']}\n"
            f"Description: {row['description']}"
        )

    def list_taxonomy_matching_keyword(self, keyword: str) -> str:
        """
        List taxonomy rows where parent or description contains a keyword.
        """
        if self.taxonomy_df is None:
            return ""

        required_cols = {"cluster", "parent", "description"}

        if not required_cols.issubset(self.taxonomy_df.columns):
            return ""

        keyword = keyword.strip().lower()

        mask = (
            self.taxonomy_df["parent"]
            .astype(str)
            .str.lower()
            .str.contains(keyword, na=False, regex=False)
            | self.taxonomy_df["description"]
            .astype(str)
            .str.lower()
            .str.contains(keyword, na=False, regex=False)
        )

        filtered = self.taxonomy_df[mask]

        if filtered.empty:
            return ""

        filtered = filtered.sort_values("cluster")

        lines = [f"Clusters matching '{keyword}' in taxonomy:"]

        for _, row in filtered.iterrows():
            lines.append(
                f"- {row['cluster']}: parent={row['parent']}, "
                f"description={row['description']}"
            )

        return "\n".join(lines)

    def count_records_for_cluster(self, cluster_id: str) -> str:
        """
        Count records for a specific cluster.
        """
        if "cluster" not in self.df.columns:
            return "The CSV does not contain a cluster column."

        normalized_cluster_id = self._normalize_cluster_id(cluster_id)

        filtered = self.df[
            self.df["cluster"].astype(str).str.lower()
            == normalized_cluster_id.lower()
        ]

        return f"Number of records in {normalized_cluster_id}: {len(filtered)}"

    def describe_cluster(self, cluster_id: str) -> str:
        """
        Describe a cluster using taxonomy first, then record evidence.
        """
        normalized_cluster_id = self._normalize_cluster_id(cluster_id)

        taxonomy_result = self.describe_cluster_from_taxonomy(normalized_cluster_id)
        records_result = self._describe_cluster_from_records(normalized_cluster_id)

        if taxonomy_result:
            return (
                taxonomy_result
                + "\n\n"
                + "Record evidence from final_joined_output.csv:\n"
                + records_result
            )

        return records_result

    def _describe_cluster_from_records(self, cluster_id: str) -> str:
        """
        Describe a cluster using record evidence from final_joined_output.csv.
        """
        if "cluster" not in self.df.columns:
            return "The CSV does not contain a cluster column."

        normalized_cluster_id = self._normalize_cluster_id(cluster_id)

        filtered = self.df[
            self.df["cluster"].astype(str).str.lower()
            == normalized_cluster_id.lower()
        ]

        if filtered.empty:
            return f"No records found for cluster {normalized_cluster_id}."

        lines = [f"Number of records: {len(filtered)}"]

        if "cluster_name" in filtered.columns:
            cluster_names = (
                filtered["cluster_name"]
                .dropna()
                .astype(str)
                .unique()
                .tolist()
            )

            if cluster_names:
                lines.insert(0, f"Cluster name from records: {cluster_names[0]}")

        if "ata_code" in filtered.columns:
            ata_counts = (
                filtered["ata_code"]
                .dropna()
                .astype(str)
                .value_counts()
                .head(5)
            )

            if not ata_counts.empty:
                lines.append("")
                lines.append("Common ATA codes:")
                for ata_code, count in ata_counts.items():
                    lines.append(f"- {ata_code}: {count}")

        if "problem" in filtered.columns:
            sample_problems = (
                filtered["problem"]
                .dropna()
                .astype(str)
                .head(5)
                .tolist()
            )

            if sample_problems:
                lines.append("")
                lines.append("Sample problems:")
                for problem in sample_problems:
                    lines.append(f"- {self._truncate(problem, 300)}")

        if "action" in filtered.columns:
            sample_actions = (
                filtered["action"]
                .dropna()
                .astype(str)
                .head(5)
                .tolist()
            )

            if sample_actions:
                lines.append("")
                lines.append("Sample actions:")
                for action in sample_actions:
                    lines.append(f"- {self._truncate(action, 300)}")

        return "\n".join(lines)

    def show_records_for_cluster(self, cluster_id: str, limit: int = 5) -> str:
        """
        Show sample records for a cluster.
        """
        if "cluster" not in self.df.columns:
            return "The CSV does not contain a cluster column."

        normalized_cluster_id = self._normalize_cluster_id(cluster_id)

        filtered = self.df[
            self.df["cluster"].astype(str).str.lower()
            == normalized_cluster_id.lower()
        ]

        if filtered.empty:
            return f"No records found for cluster {normalized_cluster_id}."

        return self._format_records(filtered.head(limit))

    def list_cluster_names_matching(self, keyword: str) -> str:
        """
        List clusters whose taxonomy parent or description contains a keyword.
        Falls back to cluster_name search in final_joined_output.csv.
        """
        keyword = keyword.strip().lower()
        taxonomy_result = self.list_taxonomy_matching_keyword(keyword)

        if taxonomy_result:
            return taxonomy_result

        if "cluster" not in self.df.columns:
            return "The CSV does not contain a cluster column."

        if "cluster_name" not in self.df.columns:
            return "The CSV does not contain a cluster_name column."

        filtered = self.df[
            self.df["cluster_name"]
            .astype(str)
            .str.lower()
            .str.contains(keyword, na=False, regex=False)
        ]

        if filtered.empty:
            return f"No cluster names found containing keyword: {keyword}"

        cluster_df = (
            filtered[["cluster", "cluster_name"]]
            .drop_duplicates()
            .sort_values("cluster")
        )

        lines = [f"Clusters matching '{keyword}':"]

        for _, row in cluster_df.iterrows():
            lines.append(f"- {row['cluster']}: {row['cluster_name']}")

        return "\n".join(lines)

    def search_problem_action(self, keyword: str, limit: int = 5) -> str:
        """
        Search keyword in problem and action columns.
        """
        keyword = keyword.strip()

        searchable_cols = []

        if "problem" in self.df.columns:
            searchable_cols.append("problem")

        if "action" in self.df.columns:
            searchable_cols.append("action")

        if not searchable_cols:
            return "The CSV does not contain problem/action columns."

        mask = pd.Series(False, index=self.df.index)

        for col in searchable_cols:
            mask = mask | self.df[col].astype(str).str.lower().str.contains(
                keyword.lower(),
                na=False,
                regex=False,
            )

        filtered = self.df[mask]

        if filtered.empty:
            return f"No records found containing keyword: {keyword}"

        return self._format_records(filtered.head(limit))

    def search_workorder(self, workorder: str) -> str:
        """
        Search by work order number.
        """
        if "workorder" not in self.df.columns:
            return "The CSV does not contain a workorder column."

        workorder = workorder.strip()

        filtered = self.df[
            self.df["workorder"]
            .astype(str)
            .str.contains(workorder, na=False, regex=False)
        ]

        if filtered.empty:
            return f"No records found for work order: {workorder}"

        return self._format_records(filtered.head(10))

    def search_registration(self, registration: str, limit: int = 5) -> str:
        """
        Search by aircraft registration.
        """
        if "registration" not in self.df.columns:
            return "The CSV does not contain a registration column."

        registration = registration.strip().upper()

        filtered = self.df[
            self.df["registration"]
            .astype(str)
            .str.upper()
            .str.contains(registration, na=False, regex=False)
        ]

        if filtered.empty:
            return f"No records found for registration: {registration}"

        return self._format_records(filtered.head(limit))

    def search_ata_code(self, ata_code: str, limit: int = 5) -> str:
        """
        Search by ATA code.
        """
        if "ata_code" not in self.df.columns:
            return "The CSV does not contain an ata_code column."

        ata_code = ata_code.strip()

        filtered = self.df[
            self.df["ata_code"]
            .astype(str)
            .str.contains(ata_code, na=False, regex=False)
        ]

        if filtered.empty:
            return f"No records found for ATA code: {ata_code}"

        return self._format_records(filtered.head(limit))

    def lookup(self, question: str) -> str:
        """
        Main natural-language lookup router for CSV questions.
        """
        q = question.lower().strip()

        cluster_ids = self._extract_cluster_ids(q)
        workorders = self._extract_workorders(q)

        if "schema" in q or "columns" in q:
            return self.schema_summary()

        if (
            ("how many" in q or "count" in q or "number of" in q or "total" in q)
            and "record" in q
            and not cluster_ids
        ):
            return self.count_rows()

        if (
            ("how many" in q or "count" in q or "number of" in q or "total" in q)
            and "cluster" in q
            and not cluster_ids
        ):
            return self.count_clusters()

        # Category/topic cluster-list questions:
        # What are the clusters related to engine?
        # Which clusters are baffle related?
        # List clusters for flight control.
        if (
            "cluster" in q
            and (
                "related to" in q
                or "related" in q
                or "for" in q
                or "which" in q
                or "what" in q
                or "list" in q
                or "show" in q
            )
            and not cluster_ids
        ):
            keyword = self._extract_taxonomy_keyword(q)

            if keyword:
                return self.list_cluster_names_matching(keyword)

            return self.list_clusters()

        if (
            ("list" in q or "show" in q or "what are" in q or "which" in q)
            and "cluster" in q
            and not cluster_ids
        ):
            return self.list_clusters()

        if cluster_ids and (
            "how many" in q
            or "count" in q
            or "number of" in q
            or "total" in q
        ):
            return self.count_records_for_cluster(cluster_ids[0])

        if cluster_ids and (
            "show" in q
            or "list" in q
            or "records" in q
            or "record" in q
            or "examples" in q
            or "raw" in q
        ):
            return self.show_records_for_cluster(cluster_ids[0])

        if cluster_ids:
            return self.describe_cluster(cluster_ids[0])

        if workorders:
            return self.search_workorder(workorders[0])

        if "registration" in q:
            registration = self._extract_registration(q)

            if registration:
                return self.search_registration(registration)

        ata_codes = self._extract_ata_codes(q)

        if ata_codes and "ata" in q:
            return self.search_ata_code(ata_codes[0])

        keyword_search_terms = [
            "contains",
            "contain",
            "mention",
            "mentions",
            "problem contains",
            "action contains",
            "search",
            "find records",
            "show records containing",
            "records containing",
        ]

        if any(term in q for term in keyword_search_terms):
            keyword = self._extract_keyword_for_search(q)

            if keyword:
                return self.search_problem_action(keyword)

        topic_keywords = [
            "engine",
            "baffle",
            "mixture",
            "oil",
            "flight_control",
            "flight control",
            "propeller",
            "ignition",
            "intake",
            "induction",
            "valve",
            "cylinder",
            "exhaust",
            "start",
            "electrical",
            "battery",
            "tire",
            "landing gear",
            "cowling",
        ]

        if "cluster" in q:
            for keyword in topic_keywords:
                if keyword in q:
                    normalized_keyword = keyword.replace(" ", "_")
                    return self.list_cluster_names_matching(normalized_keyword)

        return (
            "CSV lookup could not confidently answer this question. "
            "Use vector retrieval for explanation-based questions."
        )

    def _format_records(self, df: pd.DataFrame) -> str:
        """
        Format records into readable text.
        """
        if df.empty:
            return "No records found."

        preferred_cols = [
            "workorder",
            "registration",
            "date_time_opened",
            "date_time_closed",
            "date",
            "ata_code",
            "cluster",
            "cluster_name",
            "problem",
            "action",
        ]

        available_cols = [col for col in preferred_cols if col in df.columns]

        if not available_cols:
            available_cols = list(df.columns[:8])

        lines = [f"Showing {len(df)} record(s):"]

        for i, (_, row) in enumerate(df.iterrows(), start=1):
            lines.append("")
            lines.append(f"Record {i}:")

            for col in available_cols:
                value = row.get(col, "")

                if pd.isna(value):
                    value = ""

                value = str(value)

                if len(value) > 500:
                    value = value[:500] + "..."

                lines.append(f"{col}: {value}")

        return "\n".join(lines)

    @staticmethod
    def _truncate(text: str, max_len: int = 300) -> str:
        text = str(text).strip()

        if len(text) <= max_len:
            return text

        return text[:max_len] + "..."

    @staticmethod
    def _normalize_cluster_id(cluster_id: str) -> str:
        """
        Normalize cluster IDs.

        Supports:
        - 22 -> c_22
        - c22 -> c_22
        - c_22 -> c_22
        """
        cluster_id = str(cluster_id).lower().strip()

        numbers = re.findall(r"\d+", cluster_id)

        if not numbers:
            return cluster_id

        return f"c_{numbers[0]}"

    @staticmethod
    def _extract_cluster_ids(text: str) -> list[str]:
        """
        Extract cluster IDs from text.

        Supports:
        - c_22
        - c22
        - cluster 22
        - cluster_22
        - cluster c_22
        - cluster_c_22
        - cluster c22
        """
        text = text.lower()

        cluster_ids = []

        # Matches c_22 or c22
        direct_matches = re.findall(r"\bc_?\d+\b", text)

        for match in direct_matches:
            number = re.findall(r"\d+", match)[0]
            cluster_ids.append(f"c_{number}")

        # Matches cluster 22, cluster_22, cluster c_22, cluster_c_22, cluster c22
        cluster_word_matches = re.findall(
            r"\bcluster[\s_]+c?_?(\d+)\b",
            text,
        )

        for number in cluster_word_matches:
            cluster_ids.append(f"c_{number}")

        unique_cluster_ids = []

        for cluster_id in cluster_ids:
            if cluster_id not in unique_cluster_ids:
                unique_cluster_ids.append(cluster_id)

        return unique_cluster_ids

    @staticmethod
    def _extract_workorders(text: str) -> list[str]:
        return re.findall(r"\b(?:wko)?\d{5,}\b", text.lower())

    @staticmethod
    def _extract_registration(text: str) -> Optional[str]:
        match = re.search(r"registration\s+([a-zA-Z0-9\-]+)", text)

        if match:
            return match.group(1)

        return None

    @staticmethod
    def _extract_ata_codes(text: str) -> list[str]:
        return re.findall(r"\b\d{4}\b", text)

    @staticmethod
    def _extract_keyword_for_search(text: str) -> Optional[str]:
        """
        Extract keyword for contains/search questions.
        """
        quoted = re.findall(r'"([^"]+)"', text)

        if quoted:
            return quoted[0]

        single_quoted = re.findall(r"'([^']+)'", text)

        if single_quoted:
            return single_quoted[0]

        patterns = [
            r"contains\s+([a-zA-Z0-9_\- /]+)",
            r"contain\s+([a-zA-Z0-9_\- /]+)",
            r"mention\s+([a-zA-Z0-9_\- /]+)",
            r"mentions\s+([a-zA-Z0-9_\- /]+)",
            r"search\s+([a-zA-Z0-9_\- /]+)",
            r"find records\s+(?:containing|with)?\s*([a-zA-Z0-9_\- /]+)",
            r"records containing\s+([a-zA-Z0-9_\- /]+)",
        ]

        for pattern in patterns:
            match = re.search(pattern, text)

            if match:
                keyword = match.group(1).strip()

                keyword = re.sub(
                    r"\b(in|from|the|csv|records|record|problem|action)$",
                    "",
                    keyword,
                ).strip()

                if keyword:
                    return keyword

        return None

    @staticmethod
    def _extract_taxonomy_keyword(text: str) -> Optional[str]:
        """
        Extract topic/category keyword for cluster-list questions.

        Examples:
        - What are the clusters related to Engine? -> engine
        - List clusters for flight control -> flight_control
        - Which clusters are baffle related? -> baffle
        """
        q = text.lower().strip()

        known_topics = [
            "engine_general",
            "engine",
            "baffle",
            "flight_control",
            "flight control",
            "oil_system",
            "oil",
            "fuel_control",
            "fuel",
            "mixture",
            "propeller",
            "ignition",
            "induction_intake",
            "induction",
            "intake",
            "valve_cover",
            "valve cover",
            "valve",
            "cylinder_exhaust",
            "cylinder",
            "exhaust",
            "start_system",
            "start",
            "electrical",
            "battery",
            "landing_gear_tire",
            "landing gear",
            "tire",
            "cowling",
            "inspection",
            "appearance_cleaning",
            "appearance",
            "cleaning",
            "pilot_reported",
            "pilot",
        ]

        for topic in known_topics:
            if topic in q:
                return topic.replace(" ", "_")

        patterns = [
            r"related to\s+([a-zA-Z0-9_\- ]+)",
            r"clusters for\s+([a-zA-Z0-9_\- ]+)",
            r"cluster for\s+([a-zA-Z0-9_\- ]+)",
            r"which clusters are\s+([a-zA-Z0-9_\- ]+)",
            r"what clusters are\s+([a-zA-Z0-9_\- ]+)",
            r"what are the clusters related to\s+([a-zA-Z0-9_\- ]+)",
        ]

        for pattern in patterns:
            match = re.search(pattern, q)

            if match:
                keyword = match.group(1).strip()
                keyword = re.sub(
                    r"\b(related|clusters|cluster|to|the|a|an|are|is)$",
                    "",
                    keyword,
                ).strip()

                if keyword:
                    return keyword.replace(" ", "_")

        return None