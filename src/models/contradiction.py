from typing import List
from pydantic import BaseModel, Field


class Contradiction(BaseModel):
    """
    A detected inconsistency between two or more predicates, typically
    sourced from different evidence documents (e.g. FIR vs witness statement).
    """

    predicate_indices: List[int] = Field(
        description="Indices into the flat predicate list involved in this contradiction"
    )
    source_documents: List[str] = Field(default_factory=list)
    contradiction_type: str = Field(
        default="factual",
        description="e.g. 'time', 'location', 'identity', 'sequence', 'factual'",
    )
    description: str = Field(description="Plain-language explanation of the contradiction")
    severity: str = Field(default="medium", description="low | medium | high")
    confidence: float = Field(default=0.5, ge=0.0, le=1.0)
