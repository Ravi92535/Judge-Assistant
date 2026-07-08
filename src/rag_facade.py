from typing import List, Optional

from langchain_chroma import Chroma
from langchain_core.documents import Document

from .factory.parse_factory import DocumentLoaderFactory
from .utils import SectionExtractor
from .chunkers import RecursiveChunker
from .processor import NationalProcessor
from .storage import VectorStoreFactory


class RAGFacade:
    """
    Ingests and queries the national-statute vector store (Bharatiya Nyaya
    Sanhita / Nagarik Suraksha Sanhita / Sakshya Adhiniyam), built entirely
    on LangChain primitives: a document loader -> SectionExtractor (custom,
    statute-structure-aware) -> RecursiveChunker (wraps
    RecursiveCharacterTextSplitter) -> langchain_chroma.Chroma.
    """

    def __init__(
        self,
        vector_store: Optional[Chroma] = None,
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
        collection_name: str = "national_law",
        persist_directory: Optional[str] = None,
    ):
        self._vector_store = vector_store or VectorStoreFactory.create(
            collection_name=collection_name,
            persist_directory=persist_directory,
        )
        self._loader_factory = DocumentLoaderFactory()
        self._processor = NationalProcessor(
            section_extractor=SectionExtractor(),
            chunker=RecursiveChunker(chunk_size=chunk_size, chunk_overlap=chunk_overlap),
        )

    def ingest_document(self, file_path: str) -> int:
        """Parse, chunk, and embed one statute PDF. Returns the number of chunks added."""
        from pathlib import Path

        filename = Path(file_path).name
        pages = self._loader_factory.load(file_path)
        chunks = self._processor.process(pages, filename)

        if not chunks:
            return 0

        ids = [chunk.metadata.get("chunk_id") or f"{filename}_{i}" for i, chunk in enumerate(chunks)]
        self._vector_store.add_documents(chunks, ids=ids)
        return len(chunks)

    def query(self, query_text: str, top_k: int = 6) -> List[Document]:
        return self._vector_store.similarity_search(query_text, k=top_k)

    def count(self) -> int:
        """Number of chunks currently stored in the vector store."""
        return self._vector_store._collection.count()  # noqa: SLF001 - Chroma has no public count()

    def as_retriever(self, top_k: int = 6):
        return self._vector_store.as_retriever(search_kwargs={"k": top_k})
