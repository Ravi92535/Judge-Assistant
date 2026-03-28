from __future__ import annotations
"""
evidence_map.py — Maps evidence, witnesses, and statements to key facts in the FIR.
"""

import json
from models.provider import get_llm_response


class EvidenceMap:
    """
    Scans the raw FIR text and produces a structured evidence map:
    - Individual evidence items with reliability notes.
    - A chronological timeline of events.
    """

    def map_evidence(self, text: str) -> dict:
        """
        Build an evidence map from the raw FIR text.

        Args:
            text: Raw text extracted from the FIR PDF.

        Returns:
            A dict with keys:
              - "evidence_items":       list of evidence objects
              - "chronological_events": ordered list of timestamped events
        """
        # BUG FIX: print was placed BEFORE the docstring in the original, which
        # made __doc__ return None. Moved below the docstring.
        print("EvidenceMap: map_evidence started")

        prompt = f"""
        Analyze the following FIR text and create an "Evidence Map".
        Identify all pieces of evidence (documents, physical items, witnesses,
        oral statements) and map them to the facts they support or disprove.

        FIR TEXT:
        {text}

        OUTPUT FORMAT (strict JSON, no markdown fences):
        {{
          "evidence_items": [
            {{
              "item": "Description of evidence / witness",
              "type": "Witness | Document | Physical | Oral Statement | Other",
              "fact_supported": "What fact does this establish?",
              "reliability_analysis": "Brief note on reliability based on text"
            }}
          ],
          "chronological_events": [
            {{
              "time": "Time of event (or 'Unknown')",
              "event": "Description of event",
              "involved_parties": ["List of names"]
            }}
          ]
        }}
        """

        response_text = get_llm_response(prompt, is_json=True, allow_fallback=False)
        return json.loads(response_text)
