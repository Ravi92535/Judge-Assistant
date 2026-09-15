# ⚖️ Judge — AI Legal Case Analysis Pipeline

> **An AI pipeline that reads raw evidence documents and produces a structured legal report: extracted facts, a timeline, contradiction detection, and applicable statute sections — grounded only in retrieved text, never hallucinated.**

> **This is decision-support tooling, not legal advice.**

---

## ✨ What It Does

Given one or more evidence files (FIRs, witness statements, court orders, scanned images), the pipeline automatically:

1. **Loads** documents — digital PDFs, scanned PDFs (auto OCR fallback via Tesseract), images, and DOCX files
2. **Chunks** evidence into small windows (600 chars) to keep the extraction LLM grounded
3. **Extracts predicates** — structured atomic facts (`subject → action → object`, with time, location, confidence) from every chunk in parallel
4. **Builds a timeline** — deterministically from extracted time fields using `python-dateutil` (no LLM, fully reproducible)
5. **Detects contradictions** — across all documents (time conflicts, location conflicts, identity conflicts, sequence conflicts)
6. **Retrieves relevant statute sections** from a local ChromaDB vector store of Indian national law (BNS / BNSS / BSA)
7. **Reasons legally** — citing only sections present in retrieved excerpts, never guessing
8. **Outputs** a fully structured `CaseReport` as JSON

---

## 🏗️ Architecture

```
Evidence Files (PDF / DOCX / Images)
        │
        ▼
┌─────────────────────────────────┐
│   DocumentLoaderFactory         │  ← Picks right loader by file type
│   + OCR auto-fallback           │  ← Scanned PDF? → Tesseract
└─────────────────────────────────┘
        │ List[LangChain Document]
        ▼
┌─────────────────────────────────┐
│   RecursiveChunker              │  ← 600-char chunks, 80-char overlap
└─────────────────────────────────┘
        │
        ▼
┌─────────────────────────────────┐
│   PredicateExtractor            │  ← LCEL chain, .batch() parallel
│   → List[Predicate]             │  ← who/did what/to whom/when/where
└─────────────────────────────────┘
        │
        ├───────────────────────────────────────┐
        ▼                                       ▼
┌────────────────────┐             ┌────────────────────────────┐
│  TimelineBuilder   │             │  ContradictionDetector     │
│  (pure Python,     │             │  (LLM, batches of 60 preds)│
│  deterministic)    │             └────────────────────────────┘
└────────────────────┘                         │
        │ List[TimelineEvent]    List[Contradiction]
        └──────────────────┬────────────────────┘
                           ▼
              ┌────────────────────────┐
              │  LegalQueryBuilder     │  ← LLM formulates retrieval query
              └────────────────────────┘
                           │
                           ▼
              ┌────────────────────────┐
              │  RAGFacade             │  ← ChromaDB similarity search
              │  (BNS / BNSS / BSA)   │  ← top-K statute chunks
              └────────────────────────┘
                           │
                           ▼
              ┌────────────────────────┐
              │  LegalReasoner         │  ← LLM reasons ONLY on retrieved
              └────────────────────────┘    statute text
                           │
                           ▼
                   ┌──────────────┐
                   │  CaseReport  │  ← Structured JSON output
                   └──────────────┘
```

---

## 📁 Project Structure

```
analyze_case.py              CLI entry point
src/
  judge_facade.py            Top-level orchestrator — wires all stages
  rag_facade.py              Statute vector store — ingest + query

  factory/
    parse_factory.py         DocumentLoaderFactory — picks loader by file type
                             with automatic OCR fallback for scanned PDFs

  parser/loaders/
    docx_loader.py           CustomDocxLoader (LangChain BaseLoader + python-docx)
    ocr_loader.py            CustomOcrLoader (Tesseract + pdf2image)

  chunkers/                  RecursiveChunker (wraps RecursiveCharacterTextSplitter)
  processor/                 EvidenceProcessor (flat) / NationalProcessor (chapter-aware)

  extraction/
    predicate_extractor.py   LCEL chain — extracts facts via .batch()
    prompts.py               All 4 LLM prompt templates
    schemas.py               PredicateBatch schema

  analysis/
    timeline_builder.py      Deterministic timeline — pure Python, no LLM
    contradiction_detector.py LLM contradiction detection in batches of 60

  legal/
    query_builder.py         LegalQueryBuilder — formulates RAG retrieval query
    legal_reasoner.py        LegalReasoner — grounded legal analysis (BNS/BNSS/BSA)
    summarizers.py           Helper functions to format predicates/timeline/contradictions
    schemas.py               LegalQueryOutput, LegalReasoningOutput

  llm/                       LLMFactory → ChatGroq / ChatGoogleGenerativeAI
  embeddings/                EmbeddingFactory → HuggingFaceEmbeddings / OpenAIEmbeddings
  storage/                   VectorStoreFactory → langchain_chroma.Chroma
  utils/                     SectionExtractor — statute-structure-aware PDF parser

  models/
    predicate.py             Predicate (subject/predicate/object/time/location/confidence)
    case_report.py           CaseReport — final pipeline output
    contradiction.py         Contradiction (type/severity/indices/description)
    timeline_event.py        TimelineEvent (raw_time/parsed_time/is_sortable)
    legal_analysis.py        LegalAnalysis + ApplicableProvision

  enums/                     LLMProvider, ParserType, SourceType

NationalDocs/                Indian statute PDFs (BNS, BNSS, BSA)
Evidence/                    Put your case evidence files here
chroma_db/                   Persisted ChromaDB vector store (auto-created)
report.json                  Sample output
```

---

## 🚀 Quick Start

**Requirements:** Python 3.12+, [uv](https://docs.astral.sh/uv/)

```bash
# Install dependencies
uv sync

# For OCR on scanned PDFs / images (macOS)
brew install tesseract poppler

# Configure API keys
cp .env.example .env   # then edit .env with your keys
```

```bash
# Ingest statutes + analyze evidence in one command
uv run analyze_case.py Evidence/*.pdf --output report.json

# With more statute chunks retrieved (default is 6)
uv run analyze_case.py Evidence/fir.pdf --output report.json --top-k 10
```

---

## 🐍 Python API

```python
from src import JudgeFacade

judge = JudgeFacade()

# Ingest national statutes — skip if chroma_db/ already exists
judge.ingest_national_law([
    "NationalDocs/THE BHARATIYA NYAYA SANHITA, 2023.pdf",
    "NationalDocs/THE BHARATIYA NAGARIK SURAKSHA SANHITA, 2023.pdf",
    "NationalDocs/THE BHARATIYA SAKSHYA ADHINIYAM, 2023.pdf",
])

# Analyze a case — supports mixed formats
report = judge.analyze_case([
    "Evidence/fir.pdf",
    "Evidence/witness_statement.docx",
    "Evidence/scene_photo.jpg",
])

print(report.model_dump_json(indent=2))
```

---

## 🗂️ Output Format (`CaseReport`)

```json
{
  "evidence_documents": ["fir.pdf", "witness_statement.docx"],
  "predicates": [
    {
      "subject": "Accused",
      "predicate": "entered",
      "object": "complainant's house",
      "time": "11 PM on 3rd March 2024",
      "location": "Sector 12, Delhi",
      "source_document": "fir.pdf",
      "confidence": 0.92
    }
  ],
  "timeline": [
    {
      "raw_time": "11 PM on 3rd March 2024",
      "parsed_time": "2024-03-03T23:00:00",
      "description": "Accused entered complainant's house at 11 PM on 3rd March 2024",
      "is_sortable": true
    }
  ],
  "contradictions": [
    {
      "predicate_indices": [2, 7],
      "source_documents": ["fir.pdf", "witness_statement.docx"],
      "contradiction_type": "time",
      "description": "FIR states entry at 11 PM; witness states entry at 9 PM",
      "severity": "high",
      "confidence": 0.87
    }
  ],
  "legal_analysis": {
    "case_summary": "...",
    "legal_query": "...",
    "applicable_provisions": [
      {
        "act": "Bharatiya Nyaya Sanhita, 2023",
        "section": "Section 329",
        "title": "House-trespass",
        "reasoning": "The accused unlawfully entered...",
        "supporting_source_chunk_ids": ["bns_chunk_142"],
        "confidence": 0.84
      }
    ],
    "contested_facts_considered": ["Time of entry is disputed between FIR and witness"],
    "caveats": ["This is an automated decision-support summary, not legal advice."]
  }
}
```

---

## ⚙️ Configuration

Create a `.env` file:

```env
# LLM — choose groq (default) or gemini
LLM_PROVIDER=groq
GROQ_API_KEY=your_groq_key
GROQ_MODEL=llama-3.3-70b-versatile        # optional, this is the default

# GEMINI_API_KEY=your_gemini_key
# GEMINI_MODEL=gemini-2.0-flash           # optional, this is the default

# Embeddings — choose huggingface (default, runs locally) or openai
EMBEDDING_PROVIDER=huggingface
EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2   # optional

# OpenAI embeddings (if EMBEDDING_PROVIDER=openai)
# OPENAI_API_KEY=your_openai_key
# EMBEDDING_MODEL=text-embedding-3-small

# Vector store persistence directory
# CHROMA_DIR=chroma_db

# LangSmith tracing (optional)
# LANGCHAIN_TRACING_V2=true
# LANGSMITH_API_KEY=your_langsmith_key
# LANGSMITH_PROJECT=judge
```

---

## 📄 Supported Input Formats

| Format | Loader | Notes |
|---|---|---|
| `.pdf` (digital) | `PyPDFLoader` | LangChain community loader |
| `.pdf` (scanned) | `CustomOcrLoader` | Auto-fallback if < 40 chars extracted |
| `.docx` / `.doc` | `CustomDocxLoader` | python-docx based |
| `.png` `.jpg` `.jpeg` | `CustomOcrLoader` | Tesseract + Pillow |
| `.tif` `.tiff` `.bmp` | `CustomOcrLoader` | Tesseract + Pillow |

---

## 🧠 Hallucination Prevention

Three layers keep the LLM honest:

| Layer | Mechanism |
|---|---|
| **Small chunks** | 600-char evidence windows — LLM only sees a bounded slice of text |
| **Strict prompts** | "Do NOT invent facts not stated above. Return empty list if no facts." |
| **Retrieval grounding** | Legal reasoning cites only sections retrieved from ChromaDB — never guesses |
| **Deterministic timeline** | `python-dateutil`, not the LLM — no hallucinated dates |

---

## ⚡ Concurrency

| Stage | Method | Workers |
|---|---|---|
| Document loading | `ThreadPoolExecutor` | 4 threads |
| Predicate extraction | LangChain `.batch()` | 6 concurrent LLM calls |
| Contradiction detection | LangChain `.batch()` | 4 concurrent LLM calls |

All LLM batch calls use `return_exceptions=True` — one failing document/chunk never crashes the pipeline.

---

## 📚 National Law Corpus

| Statute | Replaces |
|---|---|
| Bharatiya Nyaya Sanhita (BNS) 2023 | Indian Penal Code 1860 |
| Bharatiya Nagarik Suraksha Sanhita (BNSS) 2023 | Code of Criminal Procedure 1973 |
| Bharatiya Sakshya Adhiniyam (BSA) 2023 | Indian Evidence Act 1872 |

Ingested once into `chroma_db/` and reused across all case analyses. Statute chunks respect section boundaries (handled by the custom `SectionExtractor`).

---

## 🧪 Tests

```bash
# Sanity check — RAG ingestion + retrieval
uv run test/test_rag_facade.py
```

---

## 📦 Key Dependencies

| Library | Purpose |
|---|---|
| `langchain` / `langchain-core` | LCEL chains, document model |
| `langchain-chroma` | ChromaDB integration |
| `langchain-huggingface` | Local sentence-transformer embeddings |
| `langchain-groq` | Groq LLM |
| `langchain-google-genai` | Gemini LLM |
| `chromadb` | Local vector database |
| `pydantic v2` | All data schemas |
| `pypdf` | Digital PDF parsing |
| `pytesseract` + `pdf2image` | OCR for scanned documents |
| `python-docx` | Word document parsing |
| `python-dateutil` | Fuzzy date parsing for timeline |
| `langsmith` | Optional LangChain tracing |

---

