#!/usr/bin/env bash
# Start / stop / status for Phase 1 Streamlit review dashboard.
# Uses a supervisor loop so crashes auto-restart (Cursor shells often kill bare children).
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
PORT="${REVIEW_PORT:-8501}"
PY="${REVIEW_PYTHON:-$HOME/.venvs/soccer-rfdetr312/bin/python3}"
LOGDIR="$ROOT/reports/eval_match3/improve_eng_loop"
LOG="$LOGDIR/streamlit_review.log"
PIDFILE="$LOGDIR/streamlit_review.pid"
SUPERVISOR_PIDFILE="$LOGDIR/streamlit_supervisor.pid"
SUPERVISOR="$ROOT/scripts/review_dashboard_supervisor.sh"
LAUNCHD_LABEL="com.soccercoach.review-dashboard"

mkdir -p "$LOGDIR"
chmod +x "$SUPERVISOR" 2>/dev/null || true

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
  curl -sf --connect-timeout 3 "http://127.0.0.1:$PORT/_stcore/health" 2>/dev/null | grep -q ok
}

launchd_loaded() {
  launchctl print "gui/$(id -u)/$LAUNCHD_LABEL" &>/dev/null
}

stop_server() {
  if launchd_loaded; then
    launchctl bootout "gui/$(id -u)/$LAUNCHD_LABEL" 2>/dev/null || true
  fi
  local pid
  pid="$(read_pid || true)"
  if pid_alive "$pid"; then
    echo "stopping supervisor pid $pid"
    kill "$pid" 2>/dev/null || true
    sleep 1
    pid_alive "$pid" && kill -9 "$pid" 2>/dev/null || true
  fi
  pid="$(cat "$SUPERVISOR_PIDFILE" 2>/dev/null || true)"
  if pid_alive "$pid"; then
    kill "$pid" 2>/dev/null || true
    sleep 1
    pid_alive "$pid" && kill -9 "$pid" 2>/dev/null || true
  fi
  pkill -9 -f 'review_dashboard_supervisor.sh' 2>/dev/null || true
  pkill -9 -f 'streamlit run apps/review_dashboard' 2>/dev/null || true
  if lsof -iTCP:"$PORT" -sTCP:LISTEN >/dev/null 2>&1; then
    lsof -tiTCP:"$PORT" -sTCP:LISTEN | xargs kill -9 2>/dev/null || true
    sleep 1
  fi
  rm -f "$PIDFILE" "$SUPERVISOR_PIDFILE" "$LOGDIR/streamlit_child.pid"
}

status_server() {
  if launchd_loaded; then
    echo "launchd: loaded ($LAUNCHD_LABEL)"
  fi
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
    echo "healthy on port $PORT → http://127.0.0.1:$PORT/"
    exit 0
  fi
  echo "not running (port $PORT)"
  exit 1
}

start_supervisor() {
  stop_server
  cd "$ROOT"
  if [[ ! -x "$PY" ]]; then
    echo "Python not found: $PY" >&2
    echo "Set REVIEW_PYTHON or create ~/.venvs/soccer-rfdetr312" >&2
    exit 1
  fi
  if command -v setsid >/dev/null 2>&1; then
    setsid bash "$SUPERVISOR" >>"$LOGDIR/supervisor_stdout.log" 2>&1 &
  else
    nohup bash "$SUPERVISOR" >>"$LOGDIR/supervisor_stdout.log" 2>&1 &
  fi
  local spid=$!
  echo "$spid" >"$PIDFILE"
  disown "$spid" 2>/dev/null || true
  echo "supervisor pid $spid → http://127.0.0.1:$PORT/  log=$LOG"

  for i in $(seq 1 60); do
    if health_ok; then
      local realpid
      realpid="$(lsof -tiTCP:"$PORT" -sTCP:LISTEN 2>/dev/null | head -1 || true)"
      if [[ -n "$realpid" ]]; then
        echo "healthy after ${i}s (streamlit pid $realpid)"
        exit 0
      fi
    fi
    sleep 1
  done
  echo "FAILED to become healthy — see $LOG and $LOGDIR/supervisor_restarts.log" >&2
  tail -30 "$LOG" >&2 || true
  exit 1
}

start_launchd() {
  bash "$ROOT/scripts/install_review_dashboard_launchd.sh"
  for i in $(seq 1 90); do
    if health_ok; then
      echo "healthy after ${i}s (launchd)"
      exit 0
    fi
    sleep 1
  done
  echo "launchd started but health check failed — see $LOGDIR/supervisor_stderr.log" >&2
  exit 1
}

start_mac_terminal() {
  if health_ok; then
    echo "already healthy → http://127.0.0.1:$PORT/"
    exit 0
  fi
  echo "Opening Terminal.app (stays alive when Cursor closes)…"
  bash "$ROOT/scripts/open_review_dashboard_terminal.sh"
  for i in $(seq 1 90); do
    if health_ok; then
      local realpid
      realpid="$(lsof -tiTCP:"$PORT" -sTCP:LISTEN 2>/dev/null | head -1 || true)"
      echo "healthy after ${i}s → http://127.0.0.1:$PORT/ (pid ${realpid:-?})"
      exit 0
    fi
    sleep 1
  done
  echo "Terminal started but health check timed out — open http://127.0.0.1:$PORT/ in Safari/Chrome and wait ~30s" >&2
  exit 1
}

case "${1:-start}" in
  start)
    if [[ "$(uname -s)" == "Darwin" ]]; then
      start_mac_terminal
    else
      start_supervisor
    fi
    ;;
  start-bg) start_supervisor ;;
  stop) stop_server; echo "stopped" ;;
  status) status_server ;;
  restart)
    stop_server
    if [[ "$(uname -s)" == "Darwin" ]]; then
      start_mac_terminal
    else
      start_supervisor
    fi
    ;;
  terminal)
    bash "$ROOT/scripts/open_review_dashboard_terminal.sh"
    ;;
  install-launchd)
    start_launchd
    ;;
  *)
    echo "usage: $0 {start|start-bg|stop|status|restart|terminal|install-launchd}" >&2
    exit 1
    ;;
esac
