from typing import List

from langchain_core.language_models.chat_models import BaseChatModel

from ..models import Predicate, TimelineEvent, Contradiction
from ..extraction.prompts import LEGAL_QUERY_PROMPT
from .summarizers import summarize_predicates, summarize_timeline, summarize_contradictions
from .schemas import LegalQueryOutput


class LegalQueryBuilder:
    """
    LCEL chain (`prompt | llm.with_structured_output(LegalQueryOutput)`)
    that turns the case's extracted facts + timeline + contradictions into
    a single natural-language retrieval query, used to search the
    national-statute vector store for the most relevant sections.
    """

    def __init__(self, llm: BaseChatModel):
        self._chain = LEGAL_QUERY_PROMPT | llm.with_structured_output(LegalQueryOutput)

    def build(
        self,
        predicates: List[Predicate],
        timeline: List[TimelineEvent],
        contradictions: List[Contradiction],
    ) -> LegalQueryOutput:
        try:
            return self._chain.invoke(
                {
                    "predicates_summary": summarize_predicates(predicates),
                    "timeline_summary": summarize_timeline(timeline),
                    "contradictions_summary": summarize_contradictions(contradictions),
                }
            )
        except Exception:
            # Fallback so the pipeline never dead-ends on a malformed LLM reply.
            return LegalQueryOutput(
                legal_query=summarize_predicates(predicates, limit=20),
                case_summary="Case summary unavailable due to a query-generation error.",
            )
