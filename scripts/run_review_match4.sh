#!/usr/bin/env bash
# Streamlit Expert review on Match 4 batch export (5-min or growing full).
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
export PYTHONPATH="."
OUT="${1:-data/output/match_4_5min}"
export SOCCER_OUTPUT_ROOT="${OUT}"
echo "Output root: $OUT → http://127.0.0.1:8501"
streamlit run apps/review_dashboard.py --server.headless true
