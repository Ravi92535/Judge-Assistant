import os
import sys
import shutil

# Ensure project root is in python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src import RAGFacade

def main():
    pdf_path = "/Users/ravi/Downloads/Judge /NationalDocs/THE BHARATIYA NAGARIK SURAKSHA SANHITA, 2023.pdf"
    bns= "/Users/ravi/Downloads/Judge /NationalDocs/THE BHARATIYA NYAYA SANHITA, 2023.pdf"
    bsa= "/Users/ravi/Downloads/Judge /NationalDocs/THE BHARATIYA SAKSHYA ADHINIYAM, 2023.pdf"
    persist_dir = "./test_chroma_db"
    
    print("Initializing RAG Facade...")
    # Clean up old database if exists to ensure clean run
    if os.path.exists(persist_dir):
        print(f"Cleaning up old database at {persist_dir}...")
        shutil.rmtree(persist_dir)

    facade = RAGFacade(
        collection_name="test_national_docs",
        persist_directory=persist_dir
    )
    
    print(f"Ingesting document: {os.path.basename(bsa)}...")
    chunks_count = facade.ingest_document(bsa)
    print(f"Successfully ingested {chunks_count} chunks.")
    
    # Query test
    query_text = "evidence"
    print(f"\nRunning query search: '{query_text}'...")
    results = facade.query(query_text, top_k=3)
    
    print(f"Found {len(results)} matches:")
    for idx, match in enumerate(results):
        print(f"\n--- Match {idx + 1} ---")
        print(f"Metadata: {match.metadata}")
        print(f"Snippet: {match.page_content[:250]}...")
        
    assert chunks_count > 0, "No chunks were processed!"
    assert len(results) > 0, "No matching query results were found!"
    print("\nAll validation checks passed successfully!")

if __name__ == "__main__":
    main()
