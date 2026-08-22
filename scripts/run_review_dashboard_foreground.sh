#!/usr/bin/env bash
# Run review dashboard in foreground (for Terminal.app — stays alive).
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
PORT="${REVIEW_PORT:-8501}"
PY="${REVIEW_PYTHON:-$HOME/.venvs/soccer-rfdetr312/bin/python3}"
LOGDIR="$ROOT/reports/eval_match3/improve_eng_loop"
LOG="$LOGDIR/streamlit_review.log"
PIDFILE="$LOGDIR/streamlit_review.pid"

mkdir -p "$LOGDIR"
cd "$ROOT"

if [[ ! -x "$PY" ]]; then
  echo "Python not found: $PY"
  echo "Create venv: python3 -m venv ~/.venvs/soccer-rfdetr312 && pip install -r requirements.txt"
  exit 1
fi

# Stop anything on our port first
bash "$ROOT/scripts/start_review_dashboard.sh" stop 2>/dev/null || true

echo "Match Review → http://127.0.0.1:$PORT/"
echo "Log: $LOG"
echo "Press Ctrl+C to stop."
echo ""

export PYTHONPATH=.
export MPLCONFIGDIR=/tmp/mpl-soccer
exec "$PY" -m streamlit run apps/review_dashboard.py \
  --server.address 127.0.0.1 \
  --server.port "$PORT" \
  --server.headless true \
  --browser.gatherUsageStats false \
  --server.fileWatcherType none \
  2>&1 | tee -a "$LOG"
