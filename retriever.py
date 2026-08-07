"""
retriever.py

Vector retriever for the local/free cluster RAG app.

Uses:
- Chroma as vector database
- HuggingFace sentence-transformer embeddings
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional

from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings

from config import RAGConfig


DEFAULT_FREE_EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

INVALID_EMBEDDING_MODELS = {
    "text-embedding-3-small",
    "sentence-transformers/text-embedding-3-small",
}


def normalize_embedding_model(model_name: str) -> str:
    """
    Normalize invalid or empty embedding model names.

    Prevents accidentally using OpenAI embedding model names with
    HuggingFaceEmbeddings.
    """
    if not model_name:
        return DEFAULT_FREE_EMBEDDING_MODEL

    cleaned_model_name = model_name.strip()

    if cleaned_model_name in INVALID_EMBEDDING_MODELS:
        return DEFAULT_FREE_EMBEDDING_MODEL

    return cleaned_model_name


class VectorRetriever:
    """
    Chroma vector retriever using HuggingFace embeddings.
    """

    def __init__(self, config: RAGConfig):
        load_dotenv(override=True)

        self.config = config
        self.persist_directory = Path(config.chroma_persist_directory)

        self.embedding_model = normalize_embedding_model(
            getattr(config, "embedding_model", DEFAULT_FREE_EMBEDDING_MODEL)
        )

        print(f"Using retriever embedding model: {self.embedding_model}")

        self.embeddings = HuggingFaceEmbeddings(
            model_name=self.embedding_model
        )

        self.vectorstore = Chroma(
            persist_directory=str(self.persist_directory),
            embedding_function=self.embeddings,
            collection_name=config.collection_name,
        )

        self.k = getattr(config, "retrieval_k", 5)

    @staticmethod
    def build_filter(
        cluster_ids: Optional[List[str]] = None,
        parent_groups: Optional[List[str]] = None,
    ) -> Optional[Dict[str, Any]]:
        """
        Build a Chroma metadata filter scoping retrieval to clusters/groups.

        Clusters and parent groups are OR-ed: a question about the baffle group
        that also names c_38 should see both. Returns None when unscoped, which
        means "search everything".
        """
        clauses: List[Dict[str, Any]] = []

        if cluster_ids:
            clauses.append({"cluster": {"$in": list(cluster_ids)}})

        if parent_groups:
            clauses.append({"parent": {"$in": list(parent_groups)}})

        if not clauses:
            return None

        if len(clauses) == 1:
            return clauses[0]

        return {"$or": clauses}

    def retrieve(
        self,
        query: str,
        cluster_ids: Optional[List[str]] = None,
        parent_groups: Optional[List[str]] = None,
        k: Optional[int] = None,
    ):
        """
        Retrieve relevant documents, optionally scoped by cluster metadata.
        """
        where = self.build_filter(cluster_ids, parent_groups)

        return self.vectorstore.similarity_search(
            query,
            k=k or self.k,
            filter=where,
        )

    def fetch_by_metadata(
        self,
        cluster_ids: Optional[List[str]] = None,
        parent_groups: Optional[List[str]] = None,
        doc_type: Optional[str] = None,
        limit: int = 10,
    ):
        """
        Fetch documents purely by metadata, with no similarity ranking.

        Used by the metadata_search tool to pull a whole parent group's
        documents when the question is about the group as a whole.
        """
        where = self.build_filter(cluster_ids, parent_groups)

        if doc_type:
            type_clause = {"doc_type": doc_type}
            where = {"$and": [where, type_clause]} if where else type_clause

        if where is None:
            return []

        result = self.vectorstore.get(where=where, limit=limit)

        documents = []

        for content, metadata in zip(
            result.get("documents", []),
            result.get("metadatas", []),
        ):
            documents.append(Document(page_content=content, metadata=metadata or {}))

        return documents