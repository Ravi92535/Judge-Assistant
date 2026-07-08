from typing import List
from pydantic import BaseModel
from .predicate import Predicate
from .contradiction import Contradiction
from .timeline_event import TimelineEvent
from .legal_analysis import LegalAnalysis


class CaseReport(BaseModel):
    """
    The single end-to-end artifact produced by JudgeFacade.analyze_case().
    Contains every intermediate stage's output so the caller (API/UI/CLI)
    can render predicates, timeline, contradictions and the final legal
    verdict without re-running the pipeline.
    """

    evidence_documents: List[str]
    predicates: List[Predicate]
    timeline: List[TimelineEvent]
    contradictions: List[Contradiction]
    legal_analysis: LegalAnalysis
