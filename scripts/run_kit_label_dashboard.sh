#!/usr/bin/env bash
# Pre-game team kit labeling UI (Streamlit).
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
export PYTHONPATH="."
PORT="${PORT:-8503}"
echo "Kit label dashboard → http://127.0.0.1:${PORT}"
exec streamlit run apps/kit_label_dashboard.py \
  --server.headless true \
  --server.address 127.0.0.1 \
  --server.port "$PORT" \
  --browser.serverPort "$PORT" \
  --server.fileWatcherType none \
  --browser.gatherUsageStats false
