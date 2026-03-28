from __future__ import annotations
"""
app.py — Main Streamlit entry point for Judge Assistant.

Run from the judge_assistant/ directory:
    streamlit run app.py
"""

import sys
import os

# ─── BUG FIX: Ensure the judge_assistant/ directory is always on sys.path,
# regardless of the working directory the user launches streamlit from.
_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

import json
import re
from concurrent.futures import ThreadPoolExecutor, as_completed

import pdfplumber
import streamlit as st
from dotenv import load_dotenv
from tenacity import RetryError

from core.rag_service import LegalRAGService
from core.evidence_map import EvidenceMap
from core.contradiction_detector import ContradictionDetector
from core.precedent_retriever import PrecedentRetriever
from core.summarizer import legalSummarizer
from core.pdf_extractor import extract_text_with_ocr_fallback
from core.utils import retry_on_quota, delay_step
from models.provider import get_llm_response

# ─── Load environment variables ──────────────────────────────────────────────
# Look for .env two levels up (project root), then fall back to current dir.
_PROJECT_ROOT = os.path.dirname(_HERE)
load_dotenv(os.path.join(_PROJECT_ROOT, ".env"))

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")

# ─── LangSmith Tracing ───────────────────────────────────────────────────────
os.environ["LANGCHAIN_TRACING_V2"] = os.getenv("LANGCHAIN_TRACING_V2", "true")
os.environ["LANGCHAIN_ENDPOINT"] = os.getenv("LANGCHAIN_ENDPOINT", "https://api.smith.langchain.com")
os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGCHAIN_API_KEY", "")
os.environ["LANGCHAIN_PROJECT"] = os.getenv("LANGCHAIN_PROJECT", "Judge")

# ─── Page configuration ──────────────────────────────────────────────────────
st.set_page_config(
    page_title="Judge Assistant - Advanced Legal AI",
    page_icon="⚖️",
    layout="wide",
)

# ─── Custom CSS ──────────────────────────────────────────────────────────────
st.markdown(
    """
<style>
    .main { background-color: #f8f9fa; }
    .stButton>button {
        width: 100%;
        border-radius: 5px;
        height: 3em;
        background-color: #1e3a8a;
        color: white;
    }
    .report-container {
        background-color: white;
        padding: 2rem;
        border-radius: 10px;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
        border: 1px solid #e5e7eb;
    }
    h1, h2, h3 {
        color: #1e3a8a;
        font-family: 'Inter', sans-serif;
    }
    .status-box {
        padding: 1rem;
        border-radius: 5px;
        border-left: 5px solid #1e3a8a;
        background-color: #eff6ff;
        margin-bottom: 1rem;
    }
</style>
""",
    unsafe_allow_html=True,
)


# ─── Helpers ─────────────────────────────────────────────────────────────────

import pdfplumber

def extract_text_from_pdf(pdf_file) -> str:
    text_parts = []
    try:
        with pdfplumber.open(pdf_file) as pdf:
            for page in pdf.pages:
                extracted = page.extract_text()
                if extracted:
                    text_parts.append(extracted)
    except Exception as e:
        print(f"Error reading PDF: {e}")
        return ""

    return "\n".join(text_parts)



def generate_legal_report(text: str) -> dict | None:
    """
    Sends the FIR text to the centralized LLM provider and returns a
    structured JSON dict representing the case.
    """
    prompt = """
    You are an expert Indian Legal Assistant. Your task is to analyze the following
    First Information Report (FIR) text and convert it into a strictly structured
    JSON format.

    The JSON must follow this schema exactly:
    {
      "case_identification": {
        "fir_no": "Extract or null",
        "district": "Extract or null",
        "police_station": "Extract or null",
        "occurrence_date_time": "Extract or null",
        "reported_date_time": "Extract or null"
      },
      "complainant_details": {
        "name": "Extract or null",
        "father_husband_name": "Extract or null",
        "dob_age": "Extract or null",
        "nationality": "Extract or null",
        "occupation": "Extract or null",
        "address": "Extract or null"
      },
      "accused_details": [
        {
          "name": "Name of accused",
          "particulars": "Any other details like father's name, address if available"
        }
      ],
      "legal_provisions": {
        "sections": ["List of IPC/BNS or other sections"],
        "acts": ["List of Acts involved"]
      },
      "case_summary": {
        "brief_facts": "A concise narrative summary of the incident",
        "stolen_involved_properties": ["List of properties or null"],
        "place_of_occurrence": "Specific location mentioned"
      },
      "judge_notes": {
        "critical_observations": "Any legal nuances or critical points for the judge",
        "missing_information": "What important details are missing from the FIR"
      }
    }

    Return ONLY the raw JSON object. Do not include any markdown formatting like
    ```json ... ```. Do not include any explanatory text.

    FIR TEXT:
    {text}
    """

    try:
        response_text = get_llm_response(
            prompt.replace("{text}", text), is_json=True
        )
        # Defensive strip in case the model wraps in fences despite instructions.
        clean_json = re.sub(
            r"^```json\s*|\s*```$", "", response_text.strip(), flags=re.MULTILINE
        )
        return json.loads(clean_json)
    except Exception as e:
        st.error(f"Failed to generate legal report: {str(e)}")
        return None


# ─── Main App ─────────────────────────────────────────────────────────────────

def main():
    st.title("⚖️ Judge Assistant - Advanced Legal AI")

    # ── Sidebar ──────────────────────────────────────────────────────────────
    with st.sidebar:
        st.header("Pipeline Workflow")
        st.markdown(
            """
        ```
        Upload FIR
            ↓ JSON Extraction
            ↓ Legal RAG  ──┐
            ↓ Evidence Map ├─→ Final Summary
            ↓ Contradictions┘
        ```
        """
        )

        st.divider()
        st.header("Settings")
        if GROQ_API_KEY:
            st.success("✅ Groq API Key loaded")
        else:
            st.error("❌ Groq API Key missing")
        if GEMINI_API_KEY:
            st.success("✅ Gemini API Key loaded")

        st.divider()
        st.markdown("### System Status")

        # ─── BUG FIX: Use __file__-relative path instead of bare "judge_assistant/db_hf"
        # which broke whenever streamlit was launched from a different working directory.
        _db_path = os.path.join(_HERE, "db_hf", "chroma.sqlite3")
        if os.path.exists(_db_path):
            st.success("📚 Legal RAG DB: Ready")
        else:
            st.warning("📚 Legal RAG DB: Not initialized")

    # ── File Upload ──────────────────────────────────────────────────────────
    uploaded_file = st.file_uploader("Upload FIR PDF", type=["pdf"])

    if uploaded_file is not None:
        st.success("File uploaded successfully!")

        if st.button("🚀 Start Full Legal Pipeline"):
            with st.status("Running Legal AI Pipeline...", expanded=True) as status:

                # Step 1 — Text Extraction
                st.write("Step 1: Extracting text from PDF...")
                raw_text = extract_text_from_pdf(uploaded_file)
                if not raw_text.strip():
                    raw_text = extract_text_with_ocr_fallback(uploaded_file)

                if not raw_text.strip():
                    st.error("Text extraction failed — the PDF may be scanned/image-only.")
                    return
                
                # Step 2 — Structured JSON
                st.write("Step 2: Generating structured case data...")
                case_json = generate_legal_report(raw_text)
                if not case_json:
                    return
                

                # Step 3 — Initialize modules
                st.write("Step 3: Initializing analysis modules...")
                rag = LegalRAGService()
                try:
                    if not rag.load_vector_store():
                        st.info(
                            "RAG Database not found. Initializing from Nation Docs "
                            "(this may take a while)..."
                        )
                        rag.ingest_documents()
                except Exception as e:
                    st.warning(
                        f"RAG Database issue: {str(e)}. "
                        "Precedent retrieval might be limited."
                    )

                e_map = EvidenceMap()
                c_detector = ContradictionDetector()
                p_retriever = PrecedentRetriever(rag)
                summarizer = legalSummarizer()

                # Step 4 — Parallel analysis
                st.write(
                    "Step 4: Running parallel legal analysis "
                    "(Evidence, Contradictions, Precedents)..."
                )

                evidence_container = st.expander(
                    "Evidence Map", expanded=True
                )
                contradiction_container = st.expander(
                    "Contradiction Detection Results", expanded=True
                )
                precedent_container = st.expander(
                    "Legal Precedents Found", expanded=True
                )

                evidence_data: dict = {}
                contradictions_data: dict = {}
                legal_refs: dict = {}

                try:
                    with ThreadPoolExecutor(max_workers=3) as executor:
                        future_evidence = executor.submit(e_map.map_evidence, raw_text)
                        future_contradictions = executor.submit(
                            c_detector.detect_contradictions, case_json, raw_text
                        )
                        future_legal_refs = executor.submit(
                            p_retriever.retrieve_references, case_json
                        )

                        futures_map = {
                            future_evidence: "evidence",
                            future_contradictions: "contradictions",
                            future_legal_refs: "precedents",
                        }

                        for future in as_completed(futures_map):
                            task_type = futures_map[future]
                            try:
                                result = future.result()
                                if task_type == "evidence":
                                    evidence_data = result
                                    evidence_container.success(
                                        "✅ Evidence Mapping Complete"
                                    )
                                    evidence_container.json(result)
                                elif task_type == "contradictions":
                                    contradictions_data = result
                                    contradiction_container.success(
                                        "✅ Contradiction Detection Complete"
                                    )
                                    contradiction_container.json(result)
                                elif task_type == "precedents":
                                    legal_refs = result
                                    precedent_container.success(
                                        "✅ Legal Reference Retrieval Complete"
                                    )
                                    precedent_container.json(result)
                            except Exception as e:
                                container = {
                                    "evidence": evidence_container,
                                    "contradictions": contradiction_container,
                                    "precedents": precedent_container,
                                }[task_type]
                                container.error(f"Error in {task_type}: {str(e)}")

                    # Step 5 — Summary
                    st.write("Step 5: Drafting final neutral summary...")
                    final_summary = summarizer.generate_summary(
                        case_json, evidence_data, contradictions_data, legal_refs
                    )

                    status.update(
                        label="Pipeline Complete!", state="complete", expanded=False
                    )

                except RetryError:
                    status.update(
                        label="API Quota Exhausted", state="error", expanded=True
                    )
                    st.error(
                        "The pipeline failed after multiple retry attempts due to "
                        "API rate limits. Please wait a few minutes and try again."
                    )
                    return
                except Exception as e:
                    status.update(
                        label="Error Occurred", state="error", expanded=True
                    )
                    st.error(f"An unexpected error occurred: {str(e)}")
                    return

                # ── Results Tabs ──────────────────────────────────────────────
                tab1, tab2, tab3, tab4, tab5 = st.tabs(
                    [
                        "📊 Structured Data",
                        "🗺️ Evidence Map",
                        "⚠️ Contradictions",
                        "⚖️ Legal References",
                        "📝 Neutral Summary",
                    ]
                )

                with tab1:
                    st.json(case_json)

                with tab2:
                    st.markdown("### Evidence Map")
                    for item in evidence_data.get("evidence_items", []):
                        with st.expander(f"{item['item']} ({item['type']})"):
                            st.write(f"**Fact Supported:** {item['fact_supported']}")
                            st.write(
                                f"**Reliability Analysis:** {item['reliability_analysis']}"
                            )
                    st.markdown("### Chronology")
                    chronology = evidence_data.get("chronological_events", [])
                    if chronology:
                        st.table(chronology)

                with tab3:
                    st.markdown("### Contradictions & Gaps")
                    for c in contradictions_data.get("contradictions", []):
                        st.markdown(
                            f"**[{c['severity']}]** {c['description']}"
                        )
                        st.caption(f"Location: {c['location_in_text']}")
                    st.markdown("### Missing Links")
                    for link in contradictions_data.get("missing_links", []):
                        st.info(link)

                with tab4:
                    st.markdown("### Relevant Acts & Sections")
                    st.write(
                        legal_refs.get("legal_report", "Analysis pending...")
                    )
                    st.markdown("#### Applied Sections")
                    for sec in legal_refs.get("relevant_sections", []):
                        with st.expander(f"Section {sec['section']}"):
                            st.write(f"**Description:** {sec['description']}")
                            st.write(
                                f"**Relevance:** {sec['relevance_to_case']}"
                            )

                with tab5:
                    st.markdown("### Neutral Case Summary")
                    st.markdown(final_summary)
                    st.download_button(
                        label="📥 Download Summary (Markdown)",
                        data=final_summary,
                        file_name=f"summary_{uploaded_file.name}.md",
                        mime="text/markdown",
                    )


if __name__ == "__main__":
    main()
    # GROQ_API_KEY = os.getenv("GROQ_API_KEY")
    # print(GROQ_API_KEY)
