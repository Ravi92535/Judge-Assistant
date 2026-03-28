"""
precedent_retriever.py — Retrieves relevant Indian legal acts and sections via RAG + LLM.
"""

import json
from .rag_service import LegalRAGService
from models.provider import get_llm_response


class PrecedentRetriever:
    """
    Uses the LegalRAGService to fetch relevant legal document chunks and then
    passes them to an LLM to produce a structured analysis of applicable acts,
    sections, and precedents.

    Args:
        rag_service: An initialised LegalRAGService instance.
    """

    def __init__(self, rag_service: LegalRAGService):
        self.rag = rag_service

    def retrieve_references(self, case_data: dict) -> dict:
        """
        Retrieve and analyse relevant legal acts and sections for the given case.

        Args:
            case_data: Structured case dict (output of the JSON extraction step).

        Returns:
            A dict with keys:
              - "applied_acts":       list of applicable act names
              - "relevant_sections":  list of section objects with descriptions
              - "legal_report":       prose summary of legal applicability
        """
        # BUG FIX: print was placed BEFORE the docstring in the original, which
        # made __doc__ return None. Moved below the docstring.
        print("PrecedentRetriever: retrieve_references started")

        sections = case_data.get("legal_provisions", {}).get("sections", [])
        acts = case_data.get("legal_provisions", {}).get("acts", [])
        summary = case_data.get("case_summary", {}).get("brief_facts", "")

        query = (
            f"Acts: {', '.join(acts)}. "
            f"Sections: {', '.join(sections)}. "
            f"Case Summary: {summary}"
        )

        # Retrieve relevant snippets from the vector store
        rag_results = self.rag.query(query, k=10)
        context = "\n".join([r.page_content for r in rag_results])

        prompt = f"""
        You are an expert Indian Legal researcher. Based on the provided case
        summary and the legal snippets retrieved from our database, identify
        the relevant acts, sections, and precedents that apply.

        CASE SUMMARY:
        {summary}

        LEGAL CONTEXT FROM DATABASE:
        {context}

        OUTPUT FORMAT (strict JSON, no markdown fences):
        {{
          "applied_acts": ["List of acts mentioned in the context that apply"],
          "relevant_sections": [
            {{
              "section": "Section number and name",
              "description": "Brief description of the section",
              "relevance_to_case": "Why this section is relevant to the facts"
            }}
          ],
          "legal_report": "A paragraph explaining which acts are used and why."
        }}
        """

        response_text = get_llm_response(prompt, is_json=True, allow_fallback=True)
        return json.loads(response_text)
