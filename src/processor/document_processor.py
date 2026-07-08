from abc import ABC, abstractmethod
from typing import List

from langchain_core.documents import Document


class DocumentProcessor(ABC):
    """Turns loader output (List[Document]) into retrieval-ready chunks (List[Document])."""

    @abstractmethod
    def process(self, documents: List[Document], filename: str) -> List[Document]:
        raise NotImplementedError
