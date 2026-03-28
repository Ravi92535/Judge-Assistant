"""
ingest_rag.py — One-time script to build the ChromaDB vector store from Nation Docs PDFs.

Run this ONCE from the judge_assistant/ directory before starting the app:

    cd Judge_v2/judge_assistant
    python ingest_rag.py

After it completes, the db_hf/ folder will contain chroma.sqlite3 and the
app's RAG sidebar status will show "Ready".
"""

import sys
import os

# Ensure imports resolve correctly regardless of launch directory
_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from core.rag_service import LegalRAGService


def main():
    rag = LegalRAGService(db_dir="src/db_hf")

    # Check if already initialised
    sqlite_path = os.path.join(rag.db_dir, "chroma.sqlite3")
    if os.path.exists(sqlite_path):
        print("\n✅ RAG vector store already exists at:", rag.db_dir)
        print("   Delete the db_hf/ folder and re-run if you want to re-ingest.\n")
        return

    # Check that Nation Docs exist
    if not os.path.isdir(rag.docs_dir):
        print(f"\n❌ Nation Docs folder not found at: {rag.docs_dir}")
        print("   Make sure the legal PDFs are in the 'Nation Docs' folder.\n")
        sys.exit(1)

    pdfs = [f for f in os.listdir(rag.docs_dir) if f.endswith(".pdf")]
    if not pdfs:
        print(f"\n❌ No PDF files found in: {rag.docs_dir}")
        sys.exit(1)

    print(f"\n📂 Found {len(pdfs)} PDF(s) in Nation Docs:")
    for pdf in pdfs:
        print(f"   • {pdf}")

    print("\n⏳ Starting ingestion — this may take 2-5 minutes on first run...\n")
    rag.ingest_documents()

    print("\n✅ RAG vector store successfully built at:", rag.db_dir)
    print("   You can now launch the app:\n")
    print("       streamlit run app.py\n")


if __name__ == "__main__":
    main()
