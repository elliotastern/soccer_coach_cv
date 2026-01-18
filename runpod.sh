#!/bin/bash
# RunPod startup script for Soccer Analysis Pipeline

# Activate virtual environment if it exists
if [ -d "venv" ]; then
    source venv/bin/activate
fi

# Start Streamlit dashboard
streamlit run src/visualization/app.py \
    --server.port 8501 \
    --server.address 0.0.0.0 \
    --server.headless true
