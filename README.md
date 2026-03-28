# Run main.py 
cd /Users/ravi/Desktop/Judge && uv run main.py

# Or run Streamlit directly
.venv/bin/python3 -m streamlit run src/app.py

cd /Users/ravi/Desktop/Judge && PYTHONPATH=. uv run src/app.py

uv run src/test_tracing.py
