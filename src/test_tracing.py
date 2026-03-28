import os
from dotenv import load_dotenv
from langsmith import traceable
# 1. Load the env file and explicitly set up LangChain traces BEFORE importing langchain modules
_HERE = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(_HERE)
load_dotenv(os.path.join(_PROJECT_ROOT, ".env"))

import sys

# 2. Import their own LLM wrapper instead of langchain_groq
sys.path.insert(0, _PROJECT_ROOT)
from src.models.provider import get_llm_response


@traceable
def test_tracing():
    print("Initializing LLM via get_llm_response...")
    
    print("Sending prompt: 'Hello, world! Say a quick greeting.'")
    try:
        response_text = get_llm_response("Hello, world! Say a quick greeting.")
        print("\n✅ Received Response:")
        print(response_text)
        print("\nDone! Please check your LangSmith dashboard to see if this trace appeared under the 'Judge' project.")
    except Exception as e:
        print(f"❌ Error calling LLM: {e}")

if __name__ == "__main__":
    # if not os.getenv("LANGCHAIN_API_KEY"):
    #     print("❌ Warning: LANGCHAIN_API_KEY is not set in your .env file!")
    #     print("Tracing will not work without it.")
        
    if not os.getenv("GROQ_API_KEY"):
        print("❌ Warning: GROQ_API_KEY is not set! The LLM call will fail.")
        
    test_tracing()
