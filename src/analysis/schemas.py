from typing import List

from pydantic import BaseModel, Field

from ..models.contradiction import Contradiction


class ContradictionBatch(BaseModel):
    """Top-level structured-output container for one contradiction-detection call."""

    contradictions: List[Contradiction] = Field(default_factory=list)
