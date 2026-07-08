from typing import Optional, Annotated
from pydantic import BaseModel, Field, field_validator


class Predicate(BaseModel):
    """
    A single atomic fact extracted from an evidence chunk by the LLM.

    Kept intentionally flat (subject/predicate/object + context) so it can be
    trivially serialized to/from JSON for LLM structured-output calls, stored
    in the vector DB metadata, or fed into contradiction/timeline analysis.
    """

    subject: Annotated[str, Field(description="Who/what the fact is about")]
    predicate: Annotated[str, Field(description="The relation/action connecting subject and object")]
    object: Annotated[str, Field(default="", description="What the subject acted on / relates to")]
    time: Annotated[
        Optional[str],
        Field(default=None, description="Date/time mentioned for this fact, in original wording if not normalizable"),
    ]
    location: Annotated[Optional[str], Field(default=None, description="Location mentioned for this fact")]
    source_document: Annotated[str, Field(description="Filename of the evidence document this fact came from")]
    confidence: Annotated[float, Field(default=0.5, ge=0.0, le=1.0)]

    # Traceability fields (not requested by the LLM, filled in by the pipeline)
    chunk_id: Optional[str] = None
    doc_id: Optional[str] = None

    @field_validator("confidence", mode="before")
    @classmethod
    def _coerce_confidence(cls, v):
        try:
            v = float(v)
        except (TypeError, ValueError):
            return 0.5
        return min(max(v, 0.0), 1.0)

    @field_validator("subject", "predicate", mode="before")
    @classmethod
    def _non_empty(cls, v):
        if v is None:
            return ""
        return str(v).strip()

    def as_fact_string(self) -> str:
        parts = [self.subject, self.predicate, self.object]
        fact = " ".join(p for p in parts if p)
        extras = []
        if self.time:
            extras.append(f"at {self.time}")
        if self.location:
            extras.append(f"in {self.location}")
        if extras:
            fact = f"{fact} ({', '.join(extras)})"
        return fact
