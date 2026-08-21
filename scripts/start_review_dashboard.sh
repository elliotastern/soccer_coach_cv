#!/usr/bin/env bash
# Start Phase 1 review dashboard and wait until healthy.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
PORT="${REVIEW_PORT:-8501}"
PY="${REVIEW_PYTHON:-$HOME/.venvs/soccer-rfdetr312/bin/python3}"
LOG="$ROOT/reports/eval_match3/improve_eng_loop/streamlit_review.log"
mkdir -p "$(dirname "$LOG")"

pkill -9 -f 'streamlit run apps/review_dashboard' 2>/dev/null || true
sleep 1
# free port if a stray holds it
if lsof -iTCP:"$PORT" -sTCP:LISTEN >/dev/null 2>&1; then
  lsof -tiTCP:"$PORT" -sTCP:LISTEN | xargs kill -9 2>/dev/null || true
  sleep 1
fi

cd "$ROOT"
nohup env PYTHONPATH=. MPLCONFIGDIR=/tmp/mpl-soccer \
  "$PY" -m streamlit run apps/review_dashboard.py \
  --server.address 127.0.0.1 \
  --server.port "$PORT" \
  --server.headless true \
  --browser.gatherUsageStats false \
  --server.fileWatcherType none \
  >"$LOG" 2>&1 &
echo "started pid $! → http://127.0.0.1:$PORT/  log=$LOG"

for i in $(seq 1 30); do
  if curl -sf --connect-timeout 1 "http://127.0.0.1:$PORT/_stcore/health" | grep -q ok; then
    echo "healthy after ${i}s"
    exit 0
  fi
  sleep 1
done
echo "FAILED to become healthy — see $LOG" >&2
tail -40 "$LOG" >&2 || true
exit 1
