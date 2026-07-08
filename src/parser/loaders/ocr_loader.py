from pathlib import Path
from typing import List

from langchain_core.document_loaders import BaseLoader
from langchain_core.documents import Document

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".tiff", ".tif", ".bmp"}


class CustomOcrLoader(BaseLoader):
    """
    LangChain-compatible loader for scanned/image evidence.

    No langchain_community loader does plain Tesseract OCR on either a
    scanned PDF or a bare image file without pulling in the much heavier
    `unstructured` dependency, so this is a small custom BaseLoader that
    does exactly that and nothing more: one Document per page (PDF) or a
    single Document (image), each tagged with a `page` metadata field so
    downstream chunks stay traceable to a specific page.
    """

    def __init__(self, file_path: str, language: str = "eng", dpi: int = 200):
        self._file_path = file_path
        self._language = language
        self._dpi = dpi

    def load(self) -> List[Document]:
        import pytesseract
        from PIL import Image

        filename = Path(self._file_path).name
        suffix = Path(self._file_path).suffix.lower()

        if suffix in IMAGE_EXTENSIONS:
            image = Image.open(self._file_path)
            text = pytesseract.image_to_string(image, lang=self._language)
            return [
                Document(
                    page_content=text,
                    metadata={"source": self._file_path, "filename": filename, "page": 0},
                )
            ]

        from pdf2image import convert_from_path

        pages = convert_from_path(self._file_path, dpi=self._dpi)
        documents = []
        for i, page_image in enumerate(pages):
            text = pytesseract.image_to_string(page_image, lang=self._language)
            documents.append(
                Document(
                    page_content=text,
                    metadata={"source": self._file_path, "filename": filename, "page": i},
                )
            )
        return documents
