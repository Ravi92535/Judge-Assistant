import os
from typing import Optional

from langchain_chroma import Chroma
from langchain_core.embeddings import Embeddings

from ..embeddings import EmbeddingFactory


class VectorStoreFactory:
    """
    Resolves a langchain_chroma.Chroma vector store. Chroma already
    implements everything a RAG pipeline needs (add_documents,
    similarity_search, as_retriever, persistence) so there's no reason
    to keep a custom VectorStore ABC wrapping it -- callers just use the
    returned Chroma instance directly.
    """

    @staticmethod
    def create(
        collection_name: str = "national_law",
        persist_directory: Optional[str] = None,
        embeddings: Optional[Embeddings] = None,
    ) -> Chroma:
        persist_directory = persist_directory or os.environ.get("CHROMA_DIR", "chroma_db")
        embeddings = embeddings or EmbeddingFactory.create()

        return Chroma(
            collection_name=collection_name,
            embedding_function=embeddings,
            persist_directory=persist_directory,
        )
