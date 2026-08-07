"""
config.py

Central configuration for the local/free cluster RAG app.

This config supports old and new field names so ingest.py,
retriever.py, graph.py, nodes.py, and csv_lookup.py can work together.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

from dotenv import load_dotenv


load_dotenv(override=True)


@dataclass
class RAGConfig:
    """
    Configuration for the cluster RAG application.
    """

    # Ollama local LLM
    ollama_model: str = os.getenv("OLLAMA_MODEL", "llama3.1:8b")
    ollama_base_url: str = os.getenv(
        "OLLAMA_BASE_URL",
        "http://localhost:11434",
    )

    # Ollama defaults to 4096, which truncates full evidence prompts. Must stay
    # comfortably above max_evidence_chars / ~4 chars-per-token.
    ollama_num_ctx: int = int(os.getenv("OLLAMA_NUM_CTX", "8192"))

    # Cap on generated answer length. Left uncapped the model writes ~4000-token
    # essays; at ~26 tok/s on an M4 that alone costs ~160s per question.
    ollama_num_predict: int = int(os.getenv("OLLAMA_NUM_PREDICT", "700"))

    # Free HuggingFace embedding model
    embedding_model: str = os.getenv(
        "EMBEDDING_MODEL",
        "sentence-transformers/all-MiniLM-L6-v2",
    )

    # Data directory
    data_dir: str = os.getenv(
        "DATA_DIR",
        os.getenv("DATA_DIRECTORY", "data"),
    )

    # Chroma vector DB directory
    db_dir: str = os.getenv(
        "DB_DIR",
        os.getenv(
            "CHROMA_PERSIST_DIRECTORY",
            os.getenv("PERSIST_DIRECTORY", "chroma_db"),
        ),
    )

    # Alias used by retriever.py
    chroma_persist_directory: str = os.getenv(
        "CHROMA_PERSIST_DIRECTORY",
        os.getenv(
            "DB_DIR",
            os.getenv("PERSIST_DIRECTORY", "chroma_db"),
        ),
    )

    # Alias used by older code
    persist_directory: str = os.getenv(
        "PERSIST_DIRECTORY",
        os.getenv(
            "DB_DIR",
            os.getenv("CHROMA_PERSIST_DIRECTORY", "chroma_db"),
        ),
    )

    # Chroma collection name
    collection_name: str = os.getenv(
        "COLLECTION_NAME",
        "cluster_maintenance_knowledge",
    )

    # Documents per vector search. The planner may issue several searches, so
    # this multiplies; 12 produced heavy overlap that the accumulator then
    # deduplicated anyway.
    retrieval_k: int = int(os.getenv("RETRIEVAL_K", "8"))

    # Chunking settings used by ingest.py
    chunk_size: int = int(os.getenv("CHUNK_SIZE", "1000"))
    chunk_overlap: int = int(os.getenv("CHUNK_OVERLAP", "200"))

    # Structured CSV lookup paths
    csv_knowledge_base_path: str = os.getenv(
        "CSV_KNOWLEDGE_BASE_PATH",
        "data/final_joined_output.csv",
    )

    cluster_taxonomy_path: str = os.getenv(
        "CLUSTER_TAXONOMY_PATH",
        "data/cluster_taxonomy.csv",
    )

    # ------------------------------------------------------------------
    # Agentic workflow limits
    # ------------------------------------------------------------------

    # Replans allowed after the first pass. 2 => at most 3 retrieval passes.
    max_iterations: int = int(os.getenv("MAX_ITERATIONS", "2"))

    # Upper bound on tool calls per plan.
    max_plan_steps: int = int(os.getenv("MAX_PLAN_STEPS", "4"))

    # Sub-questions the triage step may produce.
    max_sub_questions: int = int(os.getenv("MAX_SUB_QUESTIONS", "3"))

    # Total evidence characters kept. Bounds both context use and latency:
    # prompt processing costs roughly 1.7s per 1000 characters on an M4, and
    # the answer prompt pays it on every question.
    max_evidence_chars: int = int(os.getenv("MAX_EVIDENCE_CHARS", "8000"))

    # ------------------------------------------------------------------
    # Tavily web research (planning-only; see research.py for the contract)
    # ------------------------------------------------------------------

    tavily_api_key: str = os.getenv("TAVILY_API_KEY", "")

    enable_web_research: bool = (
        os.getenv("ENABLE_WEB_RESEARCH", "true").strip().lower()
        not in {"false", "0", "no"}
    )

    tavily_max_results: int = int(os.getenv("TAVILY_MAX_RESULTS", "3"))
    tavily_timeout_seconds: float = float(os.getenv("TAVILY_TIMEOUT_SECONDS", "8"))
    tavily_max_chars: int = int(os.getenv("TAVILY_MAX_CHARS", "4000"))
    research_max_keywords: int = int(os.getenv("RESEARCH_MAX_KEYWORDS", "12"))
    research_keyword_max_len: int = int(os.getenv("RESEARCH_KEYWORD_MAX_LEN", "40"))

    @property
    def data_directory(self) -> str:
        """
        Alias for code that expects config.data_directory.
        """
        return self.data_dir