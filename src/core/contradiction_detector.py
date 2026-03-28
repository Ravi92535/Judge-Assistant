from __future__ import annotations
"""
contradiction_detector.py — Detects internal contradictions and logical gaps in FIR data.
"""

import json
from models.provider import get_llm_response


class ContradictionDetector:
    """
    Analyzes a structured case JSON and the raw FIR text to surface
    contradictions, time/date discrepancies, and missing logical links.
    """

    def detect_contradictions(self, case_json: dict, raw_text: str) -> dict:
        """
        Detect contradictions between the structured case data and the raw FIR text.

        Args:
            case_json: Structured case dict produced by the JSON extraction step.
            raw_text:  Original raw text extracted from the FIR PDF.

        Returns:
            A dict with keys:
              - "contradictions": list of contradiction objects
              - "missing_links":  list of gap description strings
        """
        # BUG FIX: print was placed BEFORE the docstring in the original, which
        # made __doc__ return None. Moved below the docstring.
        print("ContradictionDetector: detect_contradictions started")

        prompt = f"""
        Analyze the following FIR data and raw text for contradictions,
        inconsistencies, or logical gaps.

        Look for:
        1. Time / Date discrepancies.
        2. Inconsistent witness statements.
        3. Discrepancies between the case summary and the legal provisions.
        4. Any illogical sequences of events.

        STRUCTURED CASE DATA:
        {json.dumps(case_json, indent=2)}

        RAW FIR TEXT:
        {raw_text[:5000]}

        OUTPUT FORMAT (strict JSON, no markdown fences):
        {{
          "contradictions": [
            {{
              "type": "Date/Time | Statement | Logical",
              "description": "Description of the contradiction",
              "severity": "Low | Medium | High",
              "location_in_text": "Brief snippet or page reference"
            }}
          ],
          "missing_links": [
            "Any gap where evidence does not support the claim"
          ]
        }}
        """

        response_text = get_llm_response(prompt, is_json=True, allow_fallback=False)
        return json.loads(response_text)
