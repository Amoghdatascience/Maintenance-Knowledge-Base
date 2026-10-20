"""
retriever.py

Hybrid retriever for the local/free cluster RAG app.

Uses:
- Chroma as vector database
- HuggingFace sentence-transformer embeddings
- BM25 over the same chunks (bm25.py), fused with the vector ranking by RRF

Both retrievers apply the same cluster/parent metadata scope, so hybrid search
keeps the guarantee the rest of the system depends on: a search scoped to c_38
cannot return c_37's document from either side.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings

from bm25 import BM25, reciprocal_rank_fusion, tokenize
from config import RAGConfig


DEFAULT_FREE_EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

INVALID_EMBEDDING_MODELS = {
    "text-embedding-3-small",
    "sentence-transformers/text-embedding-3-small",
}

RETRIEVAL_MODES = {"vector", "bm25", "hybrid"}


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


def document_key(content: str, metadata: Optional[Dict[str, Any]]) -> str:
    """
    Stable identity for a chunk, shared by the vector and BM25 sides.

    Chroma does not reliably hand back IDs through similarity_search across
    langchain versions, so fusion keys on source + content hash instead.
    """
    source = str((metadata or {}).get("source", ""))
    digest = hashlib.sha1(content.encode("utf-8")).hexdigest()
    return f"{source}|{digest}"


def metadata_matches(
    metadata: Optional[Dict[str, Any]],
    cluster_ids: Optional[Sequence[str]] = None,
    parent_groups: Optional[Sequence[str]] = None,
) -> bool:
    """
    Python twin of build_filter, used to scope BM25.

    Clusters and parents are OR-ed, exactly like the Chroma filter. No scope
    means everything matches.
    """
    if not cluster_ids and not parent_groups:
        return True

    metadata = metadata or {}

    if cluster_ids and metadata.get("cluster") in set(cluster_ids):
        return True

    if parent_groups and metadata.get("parent") in set(parent_groups):
        return True

    return False


class VectorRetriever:
    """
    Chroma vector retriever with optional BM25 fusion.

    The class name is kept so nothing else in the app has to change.
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

        mode = getattr(config, "retrieval_mode", "hybrid")
        self.mode = mode if mode in RETRIEVAL_MODES else "hybrid"

        self._bm25: Optional[BM25] = None
        self._bm25_documents: List[Document] = []

    # ------------------------------------------------------------------
    # BM25 index
    # ------------------------------------------------------------------

    def _ensure_bm25(self) -> Optional[BM25]:
        """
        Build the BM25 index lazily from the chunks already in Chroma.

        Reading from Chroma rather than re-chunking data/ guarantees BM25 and
        the vector side search exactly the same chunks with the same metadata.
        The index is small (hundreds of chunks), so building it takes well
        under a second.
        """
        if self._bm25 is not None:
            return self._bm25

        result = self.vectorstore.get(include=["documents", "metadatas"])
        contents = result.get("documents") or []
        metadatas = result.get("metadatas") or []

        self._bm25_documents = [
            Document(page_content=content or "", metadata=dict(metadata or {}))
            for content, metadata in zip(contents, metadatas)
        ]
        self._bm25 = BM25([tokenize(doc.page_content) for doc in self._bm25_documents])

        print(f"[retriever] BM25 index built over {len(self._bm25_documents)} chunks")

        return self._bm25

    def bm25_search(
        self,
        query: str,
        cluster_ids: Optional[List[str]] = None,
        parent_groups: Optional[List[str]] = None,
        k: Optional[int] = None,
    ) -> List[Document]:
        """
        Keyword search with the same metadata scope as the vector search.
        """
        index = self._ensure_bm25()

        if index is None or not self._bm25_documents:
            return []

        allowed = [
            position
            for position, document in enumerate(self._bm25_documents)
            if metadata_matches(document.metadata, cluster_ids, parent_groups)
        ]

        return [
            self._bm25_documents[position]
            for position, _score in index.top_n(query, k or self.k, allowed)
        ]

    # ------------------------------------------------------------------
    # Filters
    # ------------------------------------------------------------------

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

    # ------------------------------------------------------------------
    # Retrieval
    # ------------------------------------------------------------------

    def vector_search(
        self,
        query: str,
        cluster_ids: Optional[List[str]] = None,
        parent_groups: Optional[List[str]] = None,
        k: Optional[int] = None,
    ) -> List[Document]:
        """
        Plain similarity search, scoped by cluster metadata.
        """
        where = self.build_filter(cluster_ids, parent_groups)

        return self.vectorstore.similarity_search(
            query,
            k=k or self.k,
            filter=where,
        )

    def retrieve(
        self,
        query: str,
        cluster_ids: Optional[List[str]] = None,
        parent_groups: Optional[List[str]] = None,
        k: Optional[int] = None,
        mode: Optional[str] = None,
    ) -> List[Document]:
        """
        Retrieve relevant documents, optionally scoped by cluster metadata.

        hybrid: take k * candidate_multiplier from each side, fuse with RRF,
        keep the top k. Each returned document carries its vector and BM25
        ranks in metadata (copied, never mutating the store) for debugging
        and evaluation.
        """
        k = k or self.k
        mode = mode or self.mode

        if mode == "vector":
            return self.vector_search(query, cluster_ids, parent_groups, k)

        if mode == "bm25":
            return self.bm25_search(query, cluster_ids, parent_groups, k)

        pool = k * max(1, getattr(self.config, "retrieval_candidate_multiplier", 3))

        vector_docs = self.vector_search(query, cluster_ids, parent_groups, pool)

        try:
            keyword_docs = self.bm25_search(query, cluster_ids, parent_groups, pool)
        except Exception as error:  # noqa: BLE001 - BM25 must never break retrieval
            print(f"[retriever] BM25 unavailable, using vector only: {error}")
            return vector_docs[:k]

        by_key: Dict[str, Document] = {}
        vector_rank: Dict[str, int] = {}
        bm25_rank: Dict[str, int] = {}

        for rank, document in enumerate(vector_docs, start=1):
            key = document_key(document.page_content, document.metadata)
            by_key.setdefault(key, document)
            vector_rank.setdefault(key, rank)

        for rank, document in enumerate(keyword_docs, start=1):
            key = document_key(document.page_content, document.metadata)
            by_key.setdefault(key, document)
            bm25_rank.setdefault(key, rank)

        fused = reciprocal_rank_fusion(
            [list(vector_rank), list(bm25_rank)],
            k=getattr(self.config, "rrf_k", 60),
            weights=[1.0, getattr(self.config, "bm25_weight", 1.0)],
        )

        results: List[Document] = []

        for key, score in fused[:k]:
            original = by_key[key]
            metadata = dict(original.metadata or {})
            metadata["vector_rank"] = vector_rank.get(key, 0)
            metadata["bm25_rank"] = bm25_rank.get(key, 0)
            metadata["rrf_score"] = round(score, 6)
            results.append(Document(page_content=original.page_content, metadata=metadata))

        return results

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
