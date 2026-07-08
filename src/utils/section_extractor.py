import re
from typing import List

from langchain_core.documents import Document


class SectionExtractor:
    """
    Regex-based CHAPTER/Section splitter for the national statutes. There's
    no LangChain-native splitter that understands legal-document structure,
    so this stays custom -- but it now takes and returns plain
    langchain_core.documents.Document objects, with chapter/section
    metadata attached directly to each returned Document, instead of a
    bespoke Section model. That keeps Document the single, consistent
    currency the rest of the pipeline (chunking, vector storage) already
    expects.
    """

    CHAPTER_PATTERN = re.compile(
        r"CHAPTER\s+([IVXLC]+)\s*\n([A-Z][A-Z\s,&\-()]+)(.*?)(?=CHAPTER\s+[IVXLC]+\s*\n|\Z)",
        re.DOTALL,
    )
    SECTION_PATTERN = re.compile(r"(^\d+\..*?)(?=^\d+\.|\Z)", re.MULTILINE | re.DOTALL)
    SECTION_NO_PATTERN = re.compile(r"(\d+)\.")

    def extract(self, documents: List[Document], filename: str) -> List[Document]:
        """
        `documents` is the page-level output of a loader (e.g. PyPDFLoader)
        for a single statute file; they're merged into one text blob so
        chapters that span page boundaries aren't split incorrectly.
        """
        full_text = "\n".join(d.page_content for d in documents)
        section_docs: List[Document] = []

        for chapter_match in self.CHAPTER_PATTERN.finditer(full_text):
            chapter_no = chapter_match.group(1).strip()
            chapter_name = chapter_match.group(2).strip()
            chapter_text = chapter_match.group(3).strip()

            for section_match in self.SECTION_PATTERN.finditer(chapter_text):
                section_text = section_match.group(1).strip()

                sec_no_match = self.SECTION_NO_PATTERN.match(section_text)
                if not sec_no_match:
                    continue

                section_docs.append(
                    Document(
                        page_content=section_text,
                        metadata={
                            "filename": filename,
                            "chapter_no": chapter_no,
                            "chapter_name": chapter_name,
                            "section_no": sec_no_match.group(1),
                        },
                    )
                )

        return section_docs
