from typing import List

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from .chunker import Chunker


class RecursiveChunker(Chunker):
    """
    Wraps langchain_text_splitters.RecursiveCharacterTextSplitter.

    Evidence chunks are deliberately smaller than statute chunks (see
    JudgeFacade / RAGFacade construction) -- the smaller the window
    handed to the predicate-extraction LLM, the less room it has to
    hallucinate facts that aren't actually in that window.
    """

    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 200):
        self._splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )

    def split_documents(self, documents: List[Document]) -> List[Document]:
        return self._splitter.split_documents(documents)
