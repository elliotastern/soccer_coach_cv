#!/usr/bin/env bash
# Fuse mosaic emit labeling UI (Streamlit).
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
export PYTHONPATH="."
PORT="${PORT:-8502}"
echo "Fuse emit labels → http://127.0.0.1:${PORT}"
exec streamlit run apps/coach_emit_label_dashboard.py \
  --server.headless true \
  --server.address 127.0.0.1 \
  --server.port "$PORT" \
  --browser.serverPort "$PORT" \
  --server.fileWatcherType none \
  --browser.gatherUsageStats false
