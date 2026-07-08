import logging
from typing import List

from langchain_core.documents import Document
from langchain_core.language_models.chat_models import BaseChatModel

from ..models import Predicate
from .prompts import PREDICATE_EXTRACTION_PROMPT
from .schemas import PredicateBatch

logger = logging.getLogger(__name__)


class PredicateExtractor:
    """
    Runs every evidence chunk through the LLM to pull out structured
    (subject, predicate, object, time, location) facts.

    This is an LCEL chain (`prompt | llm.with_structured_output(PredicateBatch)`)
    invoked via `.batch()`, which is LangChain's native way to run many
    chunk-level calls concurrently -- it replaces what used to be a
    hand-rolled ThreadPoolExecutor, and `return_exceptions=True` gives the
    same per-chunk failure isolation the manual version had: one bad
    chunk's exception is caught and logged without sinking the batch.

    Keeping the unit of work at "one small chunk" (rather than a whole
    document) is the main hallucination guard here: the model only ever
    sees a bounded window of text and is explicitly told not to import
    facts from outside it.
    """

    def __init__(self, llm: BaseChatModel, max_concurrency: int = 6):
        self._chain = PREDICATE_EXTRACTION_PROMPT | llm.with_structured_output(PredicateBatch)
        self._max_concurrency = max_concurrency

    def extract_from_chunks(self, chunks: List[Document]) -> List[Predicate]:
        if not chunks:
            return []

        inputs = [
            {
                "source_document": chunk.metadata.get("filename", "unknown"),
                "chunk_text": chunk.page_content,
            }
            for chunk in chunks
        ]

        results = self._chain.batch(
            inputs,
            config={"max_concurrency": self._max_concurrency},
            return_exceptions=True,
        )

        predicates: List[Predicate] = []
        for chunk, result in zip(chunks, results):
            chunk_id = chunk.metadata.get("chunk_id")
            filename = chunk.metadata.get("filename", "unknown")

            if isinstance(result, Exception):
                logger.warning("Predicate extraction failed for chunk %s: %s", chunk_id, result)
                continue

            batch: PredicateBatch = result
            for extracted in batch.predicates:
                if not extracted.subject or not extracted.predicate:
                    continue
                predicates.append(
                    Predicate(
                        subject=extracted.subject,
                        predicate=extracted.predicate,
                        object=extracted.object,
                        time=extracted.time,
                        location=extracted.location,
                        source_document=filename,
                        confidence=extracted.confidence,
                        chunk_id=chunk_id,
                        doc_id=filename,
                    )
                )

        return predicates
