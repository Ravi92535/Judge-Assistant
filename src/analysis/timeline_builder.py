from typing import List

from ..models import Predicate, TimelineEvent


class TimelineBuilder:
    """
    Deterministically reconstructs a chronological timeline from the
    `time` field on extracted predicates. This is intentionally NOT an
    LLM call: date parsing/sorting is a mechanical task, and doing it in
    code keeps the timeline reproducible and free of hallucinated dates.

    Predicates whose `time` field can't be parsed into a real date are
    kept at the end of the timeline, unsorted, so no information is
    silently dropped.
    """

    def __init__(self):
        try:
            from dateutil import parser as _dateutil_parser  # noqa: F401
        except ImportError as exc:
            raise ImportError(
                "python-dateutil is required for timeline construction. "
                "Install with: pip install python-dateutil"
            ) from exc

    def build(self, predicates: List[Predicate]) -> List[TimelineEvent]:
        from dateutil import parser as dateutil_parser

        sortable: List[TimelineEvent] = []
        unsortable: List[TimelineEvent] = []

        for idx, predicate in enumerate(predicates):
            if not predicate.time:
                continue

            event = TimelineEvent(
                raw_time=predicate.time,
                description=predicate.as_fact_string(),
                source_documents=[predicate.source_document],
                predicate_indices=[idx],
            )

            try:
                event.parsed_time = dateutil_parser.parse(predicate.time, fuzzy=True)
                event.is_sortable = True
                sortable.append(event)
            except (ValueError, OverflowError):
                event.is_sortable = False
                unsortable.append(event)

        sortable.sort(key=lambda e: e.parsed_time)
        return sortable + unsortable
