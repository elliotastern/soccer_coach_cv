#!/usr/bin/env bash
# Start / stop / status for Phase 1 handover viewer (supervised serve_viewer).
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
PORT="${VIEWER_PORT:-8080}"
LOGDIR="$ROOT/reports/eval_match3/improve_eng_loop/phase1_handover"
HANDOVER_URL="http://127.0.0.1:${PORT}/phase1-handover"
SUPERVISOR="$ROOT/scripts/serve_viewer_supervisor.sh"
PIDFILE="$LOGDIR/handover_supervisor.pid"
LAUNCHD_LABEL="com.soccercoach.phase1-handover"

mkdir -p "$LOGDIR"
chmod +x "$SUPERVISOR" 2>/dev/null || true

handover_ok() {
  local code
  code="$(curl -s -m 4 -o /dev/null -w '%{http_code}' "$HANDOVER_URL" 2>/dev/null || true)"
  [[ "$code" == "302" ]] || return 1
  curl -fsS -m 4 -o /dev/null "http://127.0.0.1:$PORT/index.html" 2>/dev/null \
    && return 0
  curl -fsS -m 4 -o /dev/null \
    "http://127.0.0.1:$PORT/reports/eval_match3/improve_eng_loop/phase1_handover/index.html"
}

launchd_loaded() {
  launchctl print "gui/$(id -u)/$LAUNCHD_LABEL" &>/dev/null
}

pid_alive() {
  local pid="$1"
  [[ -n "$pid" ]] && kill -0 "$pid" 2>/dev/null
}

read_pid() {
  [[ -f "$PIDFILE" ]] && cat "$PIDFILE"
}

stop_all() {
  if launchd_loaded; then
    launchctl bootout "gui/$(id -u)/$LAUNCHD_LABEL" 2>/dev/null || true
  fi
  local pid
  pid="$(read_pid || true)"
  if pid_alive "$pid"; then
    kill "$pid" 2>/dev/null || true
    sleep 0.5
    pid_alive "$pid" && kill -9 "$pid" 2>/dev/null || true
  fi
  pkill -f 'serve_viewer_supervisor.sh' 2>/dev/null || true
  if lsof -iTCP:"$PORT" -sTCP:LISTEN >/dev/null 2>&1; then
    lsof -tiTCP:"$PORT" -sTCP:LISTEN | xargs kill -9 2>/dev/null || true
    sleep 0.5
  fi
  rm -f "$PIDFILE" "$LOGDIR/serve_viewer_supervisor.pid" "$LOGDIR/serve_viewer_child.pid"
}

status_server() {
  if launchd_loaded; then
    echo "launchd: loaded ($LAUNCHD_LABEL)"
  fi
  if handover_ok; then
    local lp
    lp="$(lsof -tiTCP:"$PORT" -sTCP:LISTEN 2>/dev/null | head -1 || true)"
    echo "healthy → $HANDOVER_URL  listener_pid=${lp:-?}"
    exit 0
  fi
  echo "not healthy (port $PORT)"
  if launchd_loaded; then
    echo "see ~/Library/Logs/soccer-coach-handover/supervisor_stderr.log"
  else
    echo "tip: bash scripts/install_phase1_handover_launchd.sh  (survives Cursor close)"
  fi
  exit 1
}

start_supervisor() {
  stop_all
  cd "$ROOT"
  python3 scripts/gold_set/build_phase1_handover_dashboard.py
  if command -v setsid >/dev/null 2>&1; then
    setsid bash "$SUPERVISOR" >>"$LOGDIR/supervisor_stdout.log" 2>&1 &
  else
    nohup bash "$SUPERVISOR" >>"$LOGDIR/supervisor_stdout.log" 2>&1 &
  fi
  echo "$!" >"$PIDFILE"
  for i in $(seq 1 40); do
    if handover_ok; then
      echo "healthy after ${i}s → $HANDOVER_URL"
      echo "note: dies when Cursor shell exits — use install-launchd for durable service"
      exit 0
    fi
    sleep 0.5
  done
  echo "handover viewer failed health check — see $LOGDIR/serve_viewer.log" >&2
  tail -20 "$LOGDIR/serve_viewer.log" 2>/dev/null || true
  exit 1
}

start_launchd() {
  bash "$ROOT/scripts/install_phase1_handover_launchd.sh"
  for i in $(seq 1 30); do
    if handover_ok; then
      echo "healthy after ${i}s (launchd) → $HANDOVER_URL"
      exit 0
    fi
    sleep 1
  done
  echo "launchd started but health check failed" >&2
  tail -20 "$HOME/Library/Logs/soccer-coach-handover/supervisor_stderr.log" 2>/dev/null || true
  exit 1
}

case "${1:-}" in
  start|start-bg) start_supervisor ;;
  stop) stop_all; echo "stopped" ;;
  status) status_server ;;
  restart) stop_all; start_supervisor ;;
  install-launchd) start_launchd ;;
  *)
    echo "usage: $0 {start|start-bg|stop|status|restart|install-launchd}" >&2
    exit 1
    ;;
esac
