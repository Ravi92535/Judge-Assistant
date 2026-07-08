from typing import Optional, List
from datetime import datetime
from pydantic import BaseModel


class TimelineEvent(BaseModel):
    """A single point on the reconstructed case timeline."""

    raw_time: Optional[str] = None
    parsed_time: Optional[datetime] = None
    description: str
    source_documents: List[str] = []
    predicate_indices: List[int] = []
    is_sortable: bool = False
