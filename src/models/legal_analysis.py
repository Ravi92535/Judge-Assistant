from typing import List, Optional
from pydantic import BaseModel, Field


class ApplicableProvision(BaseModel):
    act: str = Field(description="e.g. 'Bharatiya Nyaya Sanhita, 2023'")
    section: str = Field(description="Section number/label, e.g. 'Section 309'")
    title: Optional[str] = Field(default=None, description="Short title of the section, if known")
    reasoning: str = Field(description="Why this section applies to this case's facts")
    supporting_source_chunk_ids: List[str] = Field(default_factory=list)
    confidence: float = Field(default=0.5, ge=0.0, le=1.0)


class LegalAnalysis(BaseModel):
    """Final structured output of the legal-reasoning stage."""

    case_summary: str
    legal_query: str
    applicable_provisions: List[ApplicableProvision] = Field(default_factory=list)
    contested_facts_considered: List[str] = Field(default_factory=list)
    caveats: List[str] = Field(default_factory=list)
