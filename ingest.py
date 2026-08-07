"""
ingest.py

Object-oriented document ingestion pipeline for Agentic RAG.
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Dict, List

from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_community.document_loaders import PyPDFDirectoryLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

from config import RAGConfig
from taxonomy import load_taxonomy


class DocumentIngestor:
    """
    Handles document loading, chunking, embedding, and saving to Chroma.
    """

    def __init__(self, config: RAGConfig):
        self.config = config
        load_dotenv()

        self.taxonomy = load_taxonomy(config.cluster_taxonomy_path)

    def metadata_for(self, file_path: Path) -> Dict[str, str]:
        """
        Derive cluster/parent/doc_type metadata from a document's filename.

        Recognized shapes:
        - cluster_c_10.txt                    -> cluster c_10
        - cluster_c10_baffle_spring.txt       -> cluster c_10
        - cluster_baffle_unspecified.txt      -> parent group baffle
        - anything else                       -> reference material

        Chroma rejects None metadata values, so unknown fields are "". Note
        that an "" value will never match an $in filter, which is intended.
        """
        stem = file_path.stem

        cluster_match = re.match(r"^cluster_c_?(\d+)(?:_|$)", stem, re.IGNORECASE)

        if cluster_match:
            cluster = f"c_{cluster_match.group(1)}"

            if self.taxonomy.is_cluster(cluster):
                return {
                    "doc_type": "cluster",
                    "cluster": cluster,
                    "parent": self.taxonomy.parent_of(cluster),
                }

        group_match = re.match(r"^cluster_(.+)_unspecified$", stem)

        if group_match:
            # resolve_parent preserves canonical casing, so the capital E in
            # cluster_cylinder_Exhaust_unspecified.txt still joins correctly.
            parent = self.taxonomy.resolve_parent(group_match.group(1))

            if parent:
                return {
                    "doc_type": "parent_group",
                    "cluster": "",
                    "parent": parent,
                }

        return {"doc_type": "reference", "cluster": "", "parent": ""}

    def annotate(self, documents: List[Document]) -> List[Document]:
        """
        Attach cluster/parent/doc_type metadata to loaded documents.
        """
        for document in documents:
            source = document.metadata.get("source", "")

            if not source:
                continue

            document.metadata.update(self.metadata_for(Path(source)))

        return documents

    def load_pdf_documents(self) -> List[Document]:
        """
        Load PDF files from the data directory.
        """
        loader = PyPDFDirectoryLoader(self.config.data_dir)
        return loader.load()

    def load_text_documents(self) -> List[Document]:
        """
        Load TXT files from the data directory.
        """
        documents: List[Document] = []
        data_path = Path(self.config.data_dir)

        for file_path in data_path.glob("*.txt"):
            loader = TextLoader(str(file_path), encoding="utf-8")
            documents.extend(loader.load())

        return documents

    def load_documents(self) -> List[Document]:
        """
        Load all supported documents from the data directory.
        Currently supports PDF and TXT.
        """
        data_path = Path(self.config.data_dir)

        if not data_path.exists():
            raise FileNotFoundError(
                f"Data directory does not exist: {self.config.data_dir}"
            )

        documents: List[Document] = []

        pdf_files = list(data_path.glob("*.pdf"))
        txt_files = list(data_path.glob("*.txt"))

        if pdf_files:
            documents.extend(self.load_pdf_documents())

        if txt_files:
            documents.extend(self.load_text_documents())

        if not documents:
            raise ValueError(
                f"No supported documents found in {self.config.data_dir}. "
                "Add PDF or TXT files."
            )

        return self.annotate(documents)

    def split_documents(self, documents: List[Document]) -> List[Document]:
        """
        Split documents into chunks.
        """
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.config.chunk_size,
            chunk_overlap=self.config.chunk_overlap,
            separators=["\n\n", "\n", ".", " ", ""],
        )

        return splitter.split_documents(documents)

    def create_embeddings(self) -> HuggingFaceEmbeddings:
        """
        Create free local HuggingFace embedding model.
        """
        return HuggingFaceEmbeddings(
            model_name=self.config.embedding_model
        )

    def save_to_chroma(self, chunks: List[Document]) -> Chroma:
        """
        Save chunks to Chroma vector database.
        """
        embeddings = self.create_embeddings()

        vector_db = Chroma.from_documents(
            documents=chunks,
            embedding=embeddings,
            persist_directory=self.config.db_dir,
            collection_name=self.config.collection_name,
        )

        return vector_db

    def run(self) -> None:
        """
        Run full ingestion pipeline.

        Chroma.from_documents appends. Delete db_dir first for a clean rebuild,
        otherwise every document is indexed twice.
        """
        print(f"Loading documents from: {self.config.data_dir}")
        documents = self.load_documents()
        print(f"Loaded {len(documents)} document objects.")

        counts: dict = {}

        for document in documents:
            doc_type = document.metadata.get("doc_type", "unknown")
            counts[doc_type] = counts.get(doc_type, 0) + 1

        print(f"Document types: {counts}")

        print("Splitting documents into chunks...")
        chunks = self.split_documents(documents)
        print(f"Created {len(chunks)} chunks.")

        print(f"Saving chunks to Chroma database: {self.config.db_dir}")
        print(f"Collection: {self.config.collection_name}")
        self.save_to_chroma(chunks)

        print("Ingestion complete.")


def parse_args() -> argparse.Namespace:
    """
    Parse CLI arguments for ingestion.
    """
    parser = argparse.ArgumentParser(description="Ingest documents into Chroma.")

    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--db-dir", default="chroma_db")
    parser.add_argument("--chunk-size", type=int, default=1000)
    parser.add_argument("--chunk-overlap", type=int, default=200)
    parser.add_argument(
        "--embedding-model",
        default=RAGConfig().embedding_model,
    )

    return parser.parse_args()


def main() -> None:
    """
    CLI entry point.
    """
    args = parse_args()

    config = RAGConfig(
        data_dir=args.data_dir,
        db_dir=args.db_dir,
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap,
        embedding_model=args.embedding_model,
    )

    ingestor = DocumentIngestor(config)
    ingestor.run()


if __name__ == "__main__":
    main()