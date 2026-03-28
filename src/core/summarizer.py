"""
summarizer.py — Generates a comprehensive neutral case summary for the judge.
"""

import json
from models.provider import get_llm_response


class legalSummarizer:
    """
    Synthesises all pipeline outputs (structured case data, evidence map,
    contradictions, legal references) into a 2-3 page neutral Markdown report
    formatted for judicial review.
    """

    def generate_summary(
        self,
        case_json: dict,
        evidence_map: dict,
        contradictions: dict,
        legal_refs: dict,
    ) -> str:
        """
        Generate a comprehensive neutral summary for the judge.

        Args:
            case_json:      Structured case dict from the JSON extraction step.
            evidence_map:   Output of EvidenceMap.map_evidence().
            contradictions: Output of ContradictionDetector.detect_contradictions().
            legal_refs:     Output of PrecedentRetriever.retrieve_references().

        Returns:
            A multi-section Markdown string ready for display or download.
        """
        prompt = f"""
        You are a highly experienced Legal Consultant for the Indian Judiciary.
        Generate a comprehensive, 2-3 page neutral summary of a case based on
        the analysis components below.
        The summary must be objective, clear, and structured for a Judge to review.

        ── DATA INPUTS ────────────────────────────────────────────────────────

        1. STRUCTURED CASE DATA:
        {json.dumps(case_json, indent=2)}

        2. EVIDENCE MAP:
        {json.dumps(evidence_map, indent=2)}

        3. CONTRADICTIONS & GAPS:
        {json.dumps(contradictions, indent=2)}

        4. LEGAL REFERENCES & PRECEDENTS:
        {json.dumps(legal_refs, indent=2)}

        ── INSTRUCTIONS ───────────────────────────────────────────────────────
        - Use formal legal language.
        - Maintain a strictly neutral, objective tone.
        - Structure the report with these exact headings (Markdown ##):
            ## Executive Summary
            ## Case Background & FIR Details
            ## Legal Provisions & Acts Involved
            ## Evidence Analysis & Fact Mapping
            ## Critical Observations & Discrepancies
            ## Conclusion / Judge's Note
        - The report should be substantial (equivalent to 2-3 printed pages).

        ── OUTPUT ─────────────────────────────────────────────────────────────
        Return the full report in Markdown format only.
        """

        return get_llm_response(prompt, is_json=False)
