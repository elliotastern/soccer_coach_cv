#!/usr/bin/env bash
# Start / stop / status for Phase 1 Streamlit review dashboard.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
PORT="${REVIEW_PORT:-8501}"
PY="${REVIEW_PYTHON:-$HOME/.venvs/soccer-rfdetr312/bin/python3}"
LOGDIR="$ROOT/reports/eval_match3/improve_eng_loop"
LOG="$LOGDIR/streamlit_review.log"
PIDFILE="$LOGDIR/streamlit_review.pid"
CMD=(env PYTHONPATH=. MPLCONFIGDIR=/tmp/mpl-soccer
  "$PY" -m streamlit run apps/review_dashboard.py
  --server.address 127.0.0.1
  --server.port "$PORT"
  --server.headless true
  --browser.gatherUsageStats false
  --server.fileWatcherType none)

mkdir -p "$LOGDIR"

read_pid() {
  if [[ -f "$PIDFILE" ]]; then
    cat "$PIDFILE"
  fi
}

pid_alive() {
  local pid="$1"
  [[ -n "$pid" ]] && kill -0 "$pid" 2>/dev/null
}

health_ok() {
  curl -sf --connect-timeout 2 "http://127.0.0.1:$PORT/_stcore/health" 2>/dev/null | grep -q ok
}

stop_server() {
  local pid
  pid="$(read_pid || true)"
  if pid_alive "$pid"; then
    echo "stopping pid $pid"
    kill "$pid" 2>/dev/null || true
    sleep 1
    pid_alive "$pid" && kill -9 "$pid" 2>/dev/null || true
  fi
  pkill -9 -f 'streamlit run apps/review_dashboard' 2>/dev/null || true
  if lsof -iTCP:"$PORT" -sTCP:LISTEN >/dev/null 2>&1; then
    lsof -tiTCP:"$PORT" -sTCP:LISTEN | xargs kill -9 2>/dev/null || true
    sleep 1
  fi
  rm -f "$PIDFILE"
}

status_server() {
  local pid
  pid="$(lsof -tiTCP:"$PORT" -sTCP:LISTEN 2>/dev/null | head -1 || true)"
  if [[ -z "$pid" ]]; then
    pid="$(read_pid || true)"
  fi
  if [[ -n "$pid" ]] && pid_alive "$pid" && health_ok; then
    echo "$pid" >"$PIDFILE"
    echo "running pid=$pid → http://127.0.0.1:$PORT/  log=$LOG"
    exit 0
  fi
  if health_ok; then
    echo "healthy on port $PORT"
    exit 0
  fi
  echo "not running (port $PORT)"
  exit 1
}

start_server() {
  stop_server
  cd "$ROOT"
  if [[ ! -x "$PY" ]]; then
    echo "Python not found: $PY" >&2
    echo "Set REVIEW_PYTHON or create ~/.venvs/soccer-rfdetr312" >&2
    exit 1
  fi
  # setsid fully detaches from Cursor/agent shells (nohup alone can die on parent exit)
  if command -v setsid >/dev/null 2>&1; then
    setsid "${CMD[@]}" >>"$LOG" 2>&1 &
  else
    nohup "${CMD[@]}" >>"$LOG" 2>&1 &
  fi
  local spid=$!
  echo "$spid" >"$PIDFILE"
  disown "$spid" 2>/dev/null || true
  echo "started pid $spid → http://127.0.0.1:$PORT/  log=$LOG"

  for i in $(seq 1 45); do
    if health_ok; then
      local realpid
      realpid="$(lsof -tiTCP:"$PORT" -sTCP:LISTEN 2>/dev/null | head -1 || true)"
      if [[ -n "$realpid" ]]; then
        echo "$realpid" >"$PIDFILE"
        echo "healthy after ${i}s (listener pid $realpid)"
        exit 0
      fi
    fi
    sleep 1
  done
  echo "FAILED to become healthy — see $LOG" >&2
  tail -40 "$LOG" >&2 || true
  exit 1
}

case "${1:-start}" in
  start) start_server ;;
  stop) stop_server; echo "stopped" ;;
  status) status_server ;;
  restart) stop_server; start_server ;;
  *)
    echo "usage: $0 {start|stop|status|restart}" >&2
    exit 1
    ;;
esac
