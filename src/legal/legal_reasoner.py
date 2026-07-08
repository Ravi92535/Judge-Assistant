from typing import List

from langchain_core.documents import Document
from langchain_core.language_models.chat_models import BaseChatModel

from ..models import Predicate, TimelineEvent, Contradiction, LegalAnalysis
from ..extraction.prompts import LEGAL_REASONING_PROMPT
from .summarizers import summarize_predicates, summarize_timeline, summarize_contradictions
from .schemas import LegalReasoningOutput


class LegalReasoner:
    """
    Final stage: an LCEL chain combining the case facts/timeline/
    contradictions with the statute chunks retrieved from the national-law
    vector store (BNS / BNSS / BSA) to produce a grounded, cited list of
    applicable provisions.

    This is explicitly NOT legal advice -- it's decision support that must
    cite only sections present in the retrieved excerpts, which is why the
    prompt is instructed to return an empty list rather than guess, and
    every ApplicableProvision must list the exact chunk ids it relied on.
    """

    def __init__(self, llm: BaseChatModel):
        self._chain = LEGAL_REASONING_PROMPT | llm.with_structured_output(LegalReasoningOutput)

    def reason(
        self,
        case_summary: str,
        legal_query: str,
        predicates: List[Predicate],
        timeline: List[TimelineEvent],
        contradictions: List[Contradiction],
        retrieved_chunks: List[Document],
    ) -> LegalAnalysis:
        result: LegalReasoningOutput = self._chain.invoke(
            {
                "case_summary": case_summary,
                "legal_query": legal_query,
                "predicates_summary": summarize_predicates(predicates),
                "timeline_summary": summarize_timeline(timeline),
                "contradictions_summary": summarize_contradictions(contradictions),
                "retrieved_chunks": self._format_retrieved_chunks(retrieved_chunks),
            }
        )

        return LegalAnalysis(
            case_summary=case_summary,
            legal_query=legal_query,
            applicable_provisions=result.applicable_provisions,
            contested_facts_considered=result.contested_facts_considered,
            caveats=result.caveats + ["This is an automated decision-support summary, not legal advice."],
        )

    @staticmethod
    def _format_retrieved_chunks(retrieved_chunks: List[Document]) -> str:
        if not retrieved_chunks:
            return "(no statute excerpts retrieved)"

        lines = []
        for doc in retrieved_chunks:
            metadata = doc.metadata or {}
            chunk_id = metadata.get("chunk_id", "unknown")
            act = metadata.get("filename", "unknown act")
            chapter = metadata.get("chapter_name", "")
            section_no = metadata.get("section_no", "")
            snippet = (doc.page_content or "")[:600]
            lines.append(
                f"chunk_id={chunk_id} | act={act} | chapter={chapter} | section={section_no}\n{snippet}"
            )
        return "\n\n".join(lines)
