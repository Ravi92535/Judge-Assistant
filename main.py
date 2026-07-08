"""
Minimal smoke-test entry point for RAGFacade against the national statutes.
For the full evidence -> predicates -> contradictions -> legal-analysis
pipeline, use analyze_case.py instead.
"""

from dotenv import load_dotenv

from src import RAGFacade

NATIONAL_DOCS_DIR = "/Users/ravi/Downloads/Judge /NationalDocs"


def main():
    load_dotenv()

    pdf_path = f"{NATIONAL_DOCS_DIR}/THE BHARATIYA NAGARIK SURAKSHA SANHITA, 2023.pdf"
    bns = f"{NATIONAL_DOCS_DIR}/THE BHARATIYA NYAYA SANHITA, 2023.pdf"
    bsa = f"{NATIONAL_DOCS_DIR}/THE BHARATIYA SAKSHYA ADHINIYAM, 2023.pdf"

    rag = RAGFacade()

    if rag.count() == 0:
        print("Vector store empty — ingesting national statutes...")
        for path in (pdf_path, bns, bsa):
            n = rag.ingest_document(path)
            print(f"  ingested {n} chunks from {path}")

    results = rag.query("Which law is applicable for robbery", top_k=4)
    print("\nRelevant statute excerpts:")
    for doc in results:
        print(f"\n--- {doc.metadata.get('filename')} | chapter={doc.metadata.get('chapter_name')} | section={doc.metadata.get('section_no')} ---")
        print(doc.page_content[:300])


if __name__ == "__main__":
    main()
