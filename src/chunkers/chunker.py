from abc import ABC, abstractmethod
from typing import List

from langchain_core.documents import Document


class Chunker(ABC):
    """
    Thin abstraction over a LangChain text splitter. Operates directly on
    langchain_core.documents.Document objects (split_documents) so
    metadata set by the loader/section-extraction stage (filename,
    chapter, section, page) survives onto every resulting chunk without
    any custom Chunk/Section model needed in between.
    """

    @abstractmethod
    def split_documents(self, documents: List[Document]) -> List[Document]:
        raise NotImplementedError
