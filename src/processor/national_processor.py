from typing import List

from langchain_core.documents import Document

from ..utils import SectionExtractor
from ..chunkers import Chunker
from .document_processor import DocumentProcessor


class NationalProcessor(DocumentProcessor):
    """
    Statute-specific processing: split the loaded pages into CHAPTER/Section
    spans (SectionExtractor), then further split each span into
    embedding-sized chunks (Chunker), preserving chapter/section metadata
    on every resulting chunk so retrieval results can cite an exact
    act/chapter/section.
    """

    def __init__(self, section_extractor: SectionExtractor, chunker: Chunker):
        self._section_extractor = section_extractor
        self._chunker = chunker

    def process(self, documents: List[Document], filename: str) -> List[Document]:
        section_docs = self._section_extractor.extract(documents, filename)
        chunks = self._chunker.split_documents(section_docs)

        for idx, chunk in enumerate(chunks):
            chunk.metadata["chunk_id"] = (
                f"{filename}_ch_{chunk.metadata.get('chapter_no')}"
                f"_sec_{chunk.metadata.get('section_no')}_chunk_{idx}"
            )

        return chunks
