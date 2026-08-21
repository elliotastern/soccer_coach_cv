#!/usr/bin/env bash
set -u
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
PORT="${REVIEW_PORT:-8501}"
PY="${REVIEW_PYTHON:-$HOME/.venvs/soccer-rfdetr312/bin/python3}"
LOG="$ROOT/reports/eval_match3/improve_eng_loop/streamlit_review.log"
cd "$ROOT"
export PYTHONPATH=. MPLCONFIGDIR=/tmp/mpl-soccer
while true; do
  echo "$(date -Iseconds) starting streamlit on $PORT" >>"$LOG"
  "$PY" -m streamlit run apps/review_dashboard.py \
    --server.address 127.0.0.1 \
    --server.port "$PORT" \
    --server.headless true \
    --browser.gatherUsageStats false \
    --server.fileWatcherType none \
    >>"$LOG" 2>&1
  ec=$?
  echo "$(date -Iseconds) streamlit exited code=$ec — restart in 2s" >>"$LOG"
  sleep 2
done
