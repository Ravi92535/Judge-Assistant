from typing import List, Optional

from pydantic import BaseModel, Field


class ExtractedPredicate(BaseModel):
    """
    Structured-output schema for a single LLM extraction call. Deliberately
    narrower than the full `Predicate` domain model -- it omits
    source_document/chunk_id/doc_id, which the pipeline fills in from the
    chunk itself rather than trusting the LLM to echo them back correctly.
    """

    subject: str = Field(description="Who/what the fact is about")
    predicate: str = Field(description="The relation/action connecting subject and object")
    object: str = Field(default="", description="What the subject acted on / relates to")
    time: Optional[str] = Field(default=None, description="Date/time mentioned for this fact, or null if none")
    location: Optional[str] = Field(default=None, description="Location mentioned for this fact, or null if none")
    confidence: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        description="Confidence this fact is explicitly and unambiguously stated in the chunk",
    )


class PredicateBatch(BaseModel):
    """Top-level structured-output container -- tool-calling structured output is far more reliable targeting an object than a bare top-level array."""

    predicates: List[ExtractedPredicate] = Field(default_factory=list)
