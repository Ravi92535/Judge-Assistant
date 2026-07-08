# Judge — Legal Case Analysis Pipeline

An AI pipeline that takes raw evidence documents (FIRs, PDFs, scanned images, DOCX) and produces a structured legal report: extracted facts, a timeline, contradiction detection, and applicable statute sections — grounded only in the retrieved text, not hallucinated.

> **This is decision-support tooling, not legal advice.**

---

## What it does

Given one or more evidence files, the pipeline:

1. **Loads** documents via LangChain loaders — digital PDFs, scanned PDFs (auto OCR fallback), images, and DOCX
2. **Chunks** evidence into small windows to keep the extraction LLM grounded
3. **Extracts predicates** (subject → action → object, with time, location, confidence) from every chunk in parallel
4. **Builds a timeline** deterministically from the extracted time fields
5. **Detects contradictions** across the full predicate set
6. **Retrieves relevant statute sections** from a local Chroma vector store of Indian national law (BNS / BNSS / BSA)
7. **Reasons legally** — citing only sections present in the retrieved excerpts
8. **Outputs** a structured `CaseReport` (JSON)

---

## Project structure

```
analyze_case.py        CLI entry point
src/
  judge_facade.py      Top-level orchestrator (wires all stages)
  rag_facade.py        Statute vector store (ingest + query)
  factory/             DocumentLoaderFactory — picks loader by file type
  parser/loaders/      CustomDocxLoader, CustomOcrLoader (LangChain BaseLoader)
  chunkers/            RecursiveChunker wrapping RecursiveCharacterTextSplitter
  processor/           EvidenceProcessor (flat) / NationalProcessor (chapter-aware)
  extraction/          PredicateExtractor — LCEL chain + .batch()
  analysis/            TimelineBuilder (deterministic) + ContradictionDetector
  legal/               LegalQueryBuilder + LegalReasoner (retrieval-grounded)
  llm/                 LLMFactory → ChatGroq / ChatGoogleGenerativeAI
  embeddings/          EmbeddingFactory → HuggingFaceEmbeddings / OpenAIEmbeddings
  storage/             VectorStoreFactory → langchain_chroma.Chroma
  models/              Pydantic schemas (Predicate, CaseReport, etc.)
  enums/               LLMProvider, ParserType, SourceType
NationalDocs/          Indian statute PDFs (BNS, BNSS, BSA)
Evidence/              Put your case evidence files here
```

---

## Setup

**Requirements:** Python 3.12+, [uv](https://docs.astral.sh/uv/)

```bash
uv sync
```

For OCR support on scanned PDFs/images:

```bash
brew install tesseract poppler   # macOS
```

Copy `.env.example` to `.env` and fill in your keys (see [Configuration](#configuration) below).

---

## Running

```bash
# Analyze evidence — ingests national statutes on first run automatically
uv run analyze_case.py Evidence/*.pdf --output report.json

# Optional flags
uv run analyze_case.py Evidence/fir.pdf --output report.json --top-k 8
```

Or use the Python API directly:

```python
from src import JudgeFacade, RAGFacade

judge = JudgeFacade()

# Skip if chroma_db/ is already populated
judge.ingest_national_law([
    "NationalDocs/THE BHARATIYA NYAYA SANHITA, 2023.pdf",
    "NationalDocs/THE BHARATIYA NAGARIK SURAKSHA SANHITA, 2023.pdf",
    "NationalDocs/THE BHARATIYA SAKSHYA ADHINIYAM, 2023.pdf",
])

report = judge.analyze_case(["Evidence/fir.pdf", "Evidence/witness_statement.docx"])
print(report.model_dump_json(indent=2))
```

---

## Configuration

Create a `.env` file in the project root:

```env
# LLM — choose groq (default) or gemini
LLM_PROVIDER=groq
GROQ_API_KEY=your_groq_key
GROQ_MODEL=llama-3.3-70b-versatile      # optional, this is the default

# GEMINI_API_KEY=your_gemini_key
# GEMINI_MODEL=gemini-2.0-flash          # optional, this is the default

# Embeddings — choose huggingface (default, runs locally) or openai
EMBEDDING_PROVIDER=huggingface
EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2   # optional

# Vector store persistence directory (default: chroma_db/)
# CHROMA_DIR=chroma_db

# LangSmith tracing (optional)
# LANGCHAIN_TRACING_V2=true
# LANGSMITH_API_KEY=your_langsmith_key
# LANGSMITH_PROJECT=your_project_name
```

---

## Supported input formats

| Format | Loader |
|---|---|
| `.pdf` (digital) | `PyPDFLoader` |
| `.pdf` (scanned, no text layer) | Auto-fallback to `CustomOcrLoader` (tesseract) |
| `.docx` | `CustomDocxLoader` |
| `.png`, `.jpg`, `.bmp` | `CustomOcrLoader` (tesseract) |

---

## Sanity check

```bash
uv run test/test_rag_facade.py
```
