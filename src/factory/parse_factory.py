from pathlib import Path
from typing import List

from langchain_core.documents import Document

from ..enums import ParserType
from ..parser.loaders import CustomDocxLoader, CustomOcrLoader

MIN_EXTRACTABLE_CHARS = 40

_EXTENSION_TO_PARSER_TYPE = {
    ".pdf": ParserType.PDF,
    ".docx": ParserType.DOCX,
    ".doc": ParserType.DOCX,
    ".png": ParserType.OCR,
    ".jpg": ParserType.OCR,
    ".jpeg": ParserType.OCR,
    ".tif": ParserType.OCR,
    ".tiff": ParserType.OCR,
    ".bmp": ParserType.OCR,
}


class DocumentLoaderFactory:
    """
    Resolves the right LangChain document loader for a file and returns
    List[langchain_core.documents.Document] -- the common currency for
    every stage downstream (chunking, embedding, vector storage).

    Evidence in a real case can arrive as clean digital PDFs, scanned/
    photographed PDFs with no text layer, plain images, or Word docs --
    this factory hides that branching behind one `.load()` call,
    including an automatic OCR fallback for PDFs with no text layer.
    """

    def detect_parser_type(self, file_path: str) -> ParserType:
        suffix = Path(file_path).suffix.lower()
        return _EXTENSION_TO_PARSER_TYPE.get(suffix, ParserType.PDF)

    def load(self, file_path: str) -> List[Document]:
        parser_type = self.detect_parser_type(file_path)

        if parser_type == ParserType.PDF:
            from langchain_community.document_loaders import PyPDFLoader

            docs = PyPDFLoader(file_path).load()
            if not self._has_extractable_text(docs):
                docs = CustomOcrLoader(file_path).load()
            return self._tag_filename(docs, file_path)

        if parser_type == ParserType.DOCX:
            docs = CustomDocxLoader(file_path).load()
            return self._tag_filename(docs, file_path)

        if parser_type == ParserType.OCR:
            docs = CustomOcrLoader(file_path).load()
            return self._tag_filename(docs, file_path)

        raise ValueError(f"Unsupported file type: {file_path}")

    @staticmethod
    def _tag_filename(docs: List[Document], file_path: str) -> List[Document]:
        filename = Path(file_path).name
        for doc in docs:
            doc.metadata.setdefault("filename", filename)
            doc.metadata.setdefault("source", file_path)
        return docs

    @staticmethod
    def _has_extractable_text(docs: List[Document]) -> bool:
        total_chars = sum(len((d.page_content or "").strip()) for d in docs)
        return total_chars >= MIN_EXTRACTABLE_CHARS
