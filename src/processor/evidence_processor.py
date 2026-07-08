from typing import List

from langchain_core.documents import Document

from ..chunkers import Chunker
from .document_processor import DocumentProcessor


class EvidenceProcessor(DocumentProcessor):
    """
    Evidence documents (FIR, witness statement, charge-sheet, ...) have no
    reliable chapter/section structure the way statutes do, so this
    processor skips SectionExtractor and chunks the loaded pages directly
    via `Chunker.split_documents`, which preserves page-level metadata
    (page number, filename) onto every chunk.

    Keeping chunks small (see the smaller RecursiveChunker used by
    JudgeFacade for evidence vs. statutes) is the main hallucination guard
    for the downstream predicate-extraction LLM: the model only ever sees
    a bounded window of text and is explicitly told not to import facts
    from outside it.
    """

    def __init__(self, chunker: Chunker):
        self._chunker = chunker

    def process(self, documents: List[Document], filename: str) -> List[Document]:
        chunks = self._chunker.split_documents(documents)

        for idx, chunk in enumerate(chunks):
            chunk.metadata.setdefault("filename", filename)
            chunk.metadata["chunk_id"] = f"{filename}_chunk_{idx}"
            chunk.metadata["chunk_index"] = idx

        return chunks
