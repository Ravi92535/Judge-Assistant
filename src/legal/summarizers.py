from typing import List

from ..models import Predicate, TimelineEvent, Contradiction


def summarize_predicates(predicates: List[Predicate], limit: int = 80) -> str:
    if not predicates:
        return "(none extracted)"
    lines = [f"[{i}] {p.as_fact_string()} — source: {p.source_document}" for i, p in enumerate(predicates[:limit])]
    if len(predicates) > limit:
        lines.append(f"... and {len(predicates) - limit} more facts")
    return "\n".join(lines)


def summarize_timeline(timeline: List[TimelineEvent], limit: int = 40) -> str:
    if not timeline:
        return "(no dated events found)"
    lines = []
    for event in timeline[:limit]:
        marker = event.parsed_time.isoformat() if event.parsed_time else event.raw_time
        lines.append(f"{marker}: {event.description} ({', '.join(event.source_documents)})")
    if len(timeline) > limit:
        lines.append(f"... and {len(timeline) - limit} more events")
    return "\n".join(lines)


def summarize_contradictions(contradictions: List[Contradiction]) -> str:
    if not contradictions:
        return "(none detected)"
    lines = [
        f"- [{c.contradiction_type}/{c.severity}] {c.description} "
        f"(predicates {c.predicate_indices}, sources {c.source_documents})"
        for c in contradictions
    ]
    return "\n".join(lines)
