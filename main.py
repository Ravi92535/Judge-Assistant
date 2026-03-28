
import subprocess
import sys
from src.ingest_rag import main as ingest_rag

def main():
    # First run rag pipline of Nation Docs
    ingest_rag()

    # Automatically start the Streamlit app
    print("Starting Streamlit app...")
    try:
        subprocess.run([sys.executable, "-m", "streamlit", "run", "src/app.py"])
    except KeyboardInterrupt:
        print("\nStopping Streamlit app...")

if __name__ == '__main__':
    main()