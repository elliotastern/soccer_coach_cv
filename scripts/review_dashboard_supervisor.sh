#!/usr/bin/env bash
# Keep Streamlit review dashboard alive — restart on crash / port loss.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
PORT="${REVIEW_PORT:-8501}"
PY="${REVIEW_PYTHON:-$HOME/.venvs/soccer-rfdetr312/bin/python3}"
LOGDIR="$ROOT/reports/eval_match3/improve_eng_loop"
LOG="$LOGDIR/streamlit_review.log"
RESTART_LOG="$LOGDIR/supervisor_restarts.log"
CHILD_PIDFILE="$LOGDIR/streamlit_child.pid"
SUPERVISOR_PIDFILE="$LOGDIR/streamlit_supervisor.pid"

mkdir -p "$LOGDIR"
cd "$ROOT"

export PYTHONPATH=.
export MPLCONFIGDIR=/tmp/mpl-soccer
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-2}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-2}"

echo "$$" >"$SUPERVISOR_PIDFILE"

health_ok() {
  curl -sf --connect-timeout 3 "http://127.0.0.1:$PORT/_stcore/health" 2>/dev/null | grep -q ok
}

stop_child() {
  local pid=""
  if [[ -f "$CHILD_PIDFILE" ]]; then
    pid="$(cat "$CHILD_PIDFILE" 2>/dev/null || true)"
  fi
  if [[ -n "$pid" ]] && kill -0 "$pid" 2>/dev/null; then
    kill "$pid" 2>/dev/null || true
    sleep 1
    kill -0 "$pid" 2>/dev/null && kill -9 "$pid" 2>/dev/null || true
  fi
  if lsof -iTCP:"$PORT" -sTCP:LISTEN >/dev/null 2>&1; then
    lsof -tiTCP:"$PORT" -sTCP:LISTEN | xargs kill -9 2>/dev/null || true
    sleep 1
  fi
  rm -f "$CHILD_PIDFILE"
}

start_child() {
  if [[ ! -x "$PY" ]]; then
    echo "Python not found: $PY" >&2
    exit 1
  fi
  echo "$(date -u +%Y-%m-%dT%H:%M:%SZ) starting streamlit on :$PORT" >>"$RESTART_LOG"
  "$PY" -m streamlit run apps/review_dashboard.py \
    --server.address 127.0.0.1 \
    --server.port "$PORT" \
    --server.headless true \
    --browser.gatherUsageStats false \
    --server.fileWatcherType none \
    >>"$LOG" 2>&1 &
  local spid=$!
  echo "$spid" >"$CHILD_PIDFILE"
  disown "$spid" 2>/dev/null || true
}

trap 'stop_child; rm -f "$SUPERVISOR_PIDFILE"; exit 0' INT TERM

echo "$(date -u +%Y-%m-%dT%H:%M:%SZ) supervisor pid $$" >>"$RESTART_LOG"
start_child

while true; do
  sleep 12
  if health_ok; then
    local_pid="$(lsof -tiTCP:"$PORT" -sTCP:LISTEN 2>/dev/null | head -1 || true)"
    if [[ -n "$local_pid" ]]; then
      echo "$local_pid" >"$CHILD_PIDFILE"
    fi
    continue
  fi
  echo "$(date -u +%Y-%m-%dT%H:%M:%SZ) unhealthy — restarting" >>"$RESTART_LOG"
  stop_child
  start_child
  for _ in $(seq 1 30); do
    health_ok && break
    sleep 2
  done
done
