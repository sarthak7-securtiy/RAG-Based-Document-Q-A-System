#!/bin/bash
# Start FastAPI backend on internal port 8000 (not exposed to Cloud Run)
uvicorn backend.main:app --host 0.0.0.0 --port 8000 &

# Start Streamlit frontend on PORT (Cloud Run sets this, defaults to 8501)
streamlit run frontend/app.py \
    --server.port ${PORT:-8501} \
    --server.address 0.0.0.0 \
    --server.headless true \
    --browser.gatherUsageStats false
