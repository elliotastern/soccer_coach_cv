#!/usr/bin/env bash
# Fuse mosaic emit labeling UI (Streamlit).
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
export PYTHONPATH="."
PORT="${PORT:-8502}"
echo "Fuse emit labels → http://127.0.0.1:${PORT}"
streamlit run apps/coach_emit_label_dashboard.py \
  --server.headless true \
  --server.port "$PORT" \
  --browser.serverPort "$PORT"
