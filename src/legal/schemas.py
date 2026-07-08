from typing import List

from pydantic import BaseModel, Field

from ..models.legal_analysis import ApplicableProvision


class LegalQueryOutput(BaseModel):
    legal_query: str = Field(description="A focused retrieval query for the statute vector store")
    case_summary: str = Field(description="A one-paragraph plain-English summary of the case")


class LegalReasoningOutput(BaseModel):
    """
    Structured-output target for the final reasoning call. Deliberately
    excludes case_summary/legal_query -- those are already known (we
    generated them in the previous stage) and are merged back in code
    rather than asked of the model a second time, which avoids the model
    subtly rephrasing/drifting from what was actually used for retrieval.
    """

    applicable_provisions: List[ApplicableProvision] = Field(default_factory=list)
    contested_facts_considered: List[str] = Field(default_factory=list)
    caveats: List[str] = Field(default_factory=list)
