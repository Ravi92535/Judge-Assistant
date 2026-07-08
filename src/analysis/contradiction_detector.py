import json
import logging
from typing import List

from langchain_core.language_models.chat_models import BaseChatModel

from ..models import Predicate, Contradiction
from ..extraction.prompts import CONTRADICTION_DETECTION_PROMPT
from .schemas import ContradictionBatch

logger = logging.getLogger(__name__)

# Cap how many predicates go into a single LLM call. Large cases are split
# into independent batches so the prompt stays well within context limits;
# global predicate indices are preserved (via index_offset) so a
# contradiction still maps back to the exact predicates involved.
DEFAULT_BATCH_SIZE = 60


class ContradictionDetector:
    """
    Finds factual contradictions across (or within) evidence documents by
    handing the full, indexed predicate list to the LLM in bounded batches
    and asking it to flag conflicting facts. Batches run concurrently via
    LangChain's native `.batch()` on the LCEL chain.
    """

    def __init__(self, llm: BaseChatModel, batch_size: int = DEFAULT_BATCH_SIZE, max_concurrency: int = 4):
        self._chain = CONTRADICTION_DETECTION_PROMPT | llm.with_structured_output(ContradictionBatch)
        self._batch_size = batch_size
        self._max_concurrency = max_concurrency

    def detect(self, predicates: List[Predicate]) -> List[Contradiction]:
        if len(predicates) < 2:
            return []

        batches = [
            predicates[i : i + self._batch_size] for i in range(0, len(predicates), self._batch_size)
        ]
        offsets = [i * self._batch_size for i in range(len(batches))]

        inputs = [
            {"predicates_json": self._indexed_json(batch, offset)}
            for batch, offset in zip(batches, offsets)
        ]

        results = self._chain.batch(
            inputs,
            config={"max_concurrency": self._max_concurrency},
            return_exceptions=True,
        )

        contradictions: List[Contradiction] = []
        for result, offset in zip(results, offsets):
            if isinstance(result, Exception):
                logger.warning("Contradiction detection failed for batch at offset %s: %s", offset, result)
                continue

            batch_result: ContradictionBatch = result
            for c in batch_result.contradictions:
                # Indices from the LLM are already the global indices we
                # supplied in the prompt (see _indexed_json), so no further
                # offsetting is needed here -- just pass them through.
                contradictions.append(c)

        return contradictions

    @staticmethod
    def _indexed_json(batch: List[Predicate], offset: int) -> str:
        indexed = [
            {
                "index": offset + i,
                "subject": p.subject,
                "predicate": p.predicate,
                "object": p.object,
                "time": p.time,
                "location": p.location,
                "source_document": p.source_document,
                "confidence": p.confidence,
            }
            for i, p in enumerate(batch)
        ]
        return json.dumps(indexed, indent=2)
