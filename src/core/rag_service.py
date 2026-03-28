from __future__ import annotations
"""
rag_service.py — ChromaDB-backed legal RAG service using HuggingFace embeddings.

Ingests PDFs from the "Nation Docs" folder and provides similarity search
for relevant legal acts and sections.
"""

import os
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

# ─── Load .env from project root ─────────────────────────────────────────────
_HERE = os.path.dirname(os.path.abspath(__file__))                 # core/
_JUDGE_ASSISTANT = os.path.dirname(_HERE)                          # judge_assistant/
_PROJECT_ROOT = os.path.dirname(_JUDGE_ASSISTANT)                  # Judge_v2/
load_dotenv(os.path.join(_PROJECT_ROOT, ".env"))


class LegalRAGService:
    """
    Manages ingestion of legal PDFs into a local ChromaDB vector store
    and exposes similarity-search queries for the PrecedentRetriever.

    Args:
        docs_dir: Path (relative to project root) of the folder containing
                  legal PDF documents.
        db_dir:   Path (relative to project root) of the ChromaDB store.
    """

    def __init__(
        self,
        docs_dir: str = "Nation Docs",
        db_dir: str = "src/db_hf",
    ):
        self.docs_dir = os.path.join(_PROJECT_ROOT, docs_dir)
        self.db_dir = os.path.join(_PROJECT_ROOT, db_dir)

        self._embeddings: HuggingFaceEmbeddings | None = None
        self.vector_store: Chroma | None = None

        os.makedirs(self.db_dir, exist_ok=True)

    # ── Lazy embedding loader ─────────────────────────────────────────────────

    @property
    def embeddings(self) -> HuggingFaceEmbeddings:
        """Initialise embeddings on first access (lazy load)."""
        if self._embeddings is None:
            print(
                "Initializing HuggingFace Embeddings (all-MiniLM-L6-v2) — "
                "this may take a moment on first run..."
            )
            self._embeddings = HuggingFaceEmbeddings(
                model_name="all-MiniLM-L6-v2"
            )
        return self._embeddings

    # ── Ingestion ─────────────────────────────────────────────────────────────

    def ingest_documents(self) -> None:
        """
        Load all PDFs from docs_dir, chunk them, embed them, and persist
        the vector store to db_dir.
        """
        print(f"Loading documents from: {self.docs_dir}")
        loader = DirectoryLoader(
            self.docs_dir, glob="./*.pdf", loader_cls=PyPDFLoader
        )
        documents = loader.load()

        print(f"Splitting {len(documents)} pages into chunks...")
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000, chunk_overlap=200
        )
        chunks = text_splitter.split_documents(documents)

        print(f"Creating vector store at: {self.db_dir}")
        self.vector_store = Chroma.from_documents(
            documents=chunks,
            embedding=self.embeddings,
            persist_directory=self.db_dir,
        )
        print("Ingestion complete.")

    # ── Vector store loader ───────────────────────────────────────────────────

    def load_vector_store(self) -> bool:
        """
        Load an existing ChromaDB vector store from db_dir.

        Returns:
            True  if the store was found and loaded successfully.
            False if no store exists (caller should run ingest_documents).
        """
        sqlite_path = os.path.join(self.db_dir, "chroma.sqlite3")
        if os.path.exists(sqlite_path):
            self.vector_store = Chroma(
                persist_directory=self.db_dir,
                embedding_function=self.embeddings,
            )
            return True
        return False

    # ── Query ─────────────────────────────────────────────────────────────────

    def query(self, question: str, k: int = 5) -> list:
        """
        Retrieve the top-k most relevant document chunks for a question.

        Args:
            question: The search query string.
            k:        Number of results to return.

        Returns:
            List of LangChain Document objects.

        Raises:
            RuntimeError: If the vector store has not been initialised.
        """
        # BUG FIX: print was placed BEFORE the docstring in the original, which
        # made __doc__ return None. Moved below the docstring.
        print(f"RAGService: querying for: {question[:60]}...")

        if not self.vector_store:
            if not self.load_vector_store():
                raise RuntimeError(
                    "Vector store not initialised. "
                    "Run ingest_documents() first or ensure db_hf/ exists."
                )

        return self.vector_store.similarity_search(question, k=k)


# ─── Standalone test ─────────────────────────────────────────────────────────

if __name__ == "__main__":
    rag = LegalRAGService()
    if not rag.load_vector_store():
        rag.ingest_documents()

    test_query = "What are the sections for theft in BNS?"
    results = rag.query(test_query)
    print(f"\nTest Query: {test_query}")
    for i, res in enumerate(results):
        src = os.path.basename(res.metadata.get("source", "unknown"))
        print(f"\nResult {i + 1} (Source: {src}):\n{res.page_content[:200]}...")
