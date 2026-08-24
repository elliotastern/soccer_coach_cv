#!/usr/bin/env bash
# Keep serve_viewer alive — restart on crash / port loss (Phase 1 handover + gold tools).
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
PORT="${VIEWER_PORT:-8080}"
LOGDIR="$ROOT/reports/eval_match3/improve_eng_loop/phase1_handover"
LOG="${LOGDIR}/serve_viewer.log"
RESTART_LOG="${LOGDIR}/supervisor_restarts.log"
CHILD_PIDFILE="${LOGDIR}/serve_viewer_child.pid"
SUPERVISOR_PIDFILE="${LOGDIR}/serve_viewer_supervisor.pid"

mkdir -p "$LOGDIR"
cd "$ROOT"

echo "$$" >"$SUPERVISOR_PIDFILE"

handover_ok() {
  local code
  code="$(curl -s -m 4 -o /dev/null -w '%{http_code}' \
    "http://127.0.0.1:$PORT/phase1-handover" 2>/dev/null || true)"
  [[ "$code" == "302" ]]
}

health_ok() {
  handover_ok && curl -fsS -m 4 -o /dev/null \
    "http://127.0.0.1:$PORT/reports/eval_match3/improve_eng_loop/phase1_handover/index.html"
}

stop_child() {
  local pid=""
  if [[ -f "$CHILD_PIDFILE" ]]; then
    pid="$(cat "$CHILD_PIDFILE" 2>/dev/null || true)"
  fi
  if [[ -n "$pid" ]] && kill -0 "$pid" 2>/dev/null; then
    kill "$pid" 2>/dev/null || true
    sleep 0.5
    kill -0 "$pid" 2>/dev/null && kill -9 "$pid" 2>/dev/null || true
  fi
  if lsof -iTCP:"$PORT" -sTCP:LISTEN >/dev/null 2>&1; then
    lsof -tiTCP:"$PORT" -sTCP:LISTEN | xargs kill -9 2>/dev/null || true
    sleep 0.5
  fi
  rm -f "$CHILD_PIDFILE"
}

start_child() {
  echo "$(date -u +%Y-%m-%dT%H:%M:%SZ) starting serve_viewer :$PORT" >>"$RESTART_LOG"
  nohup python3 -u serve_viewer.py --port "$PORT" --no-fallback >>"$LOG" 2>&1 </dev/null &
  local spid=$!
  echo "$spid" >"$CHILD_PIDFILE"
  disown "$spid" 2>/dev/null || true
}

trap 'stop_child; rm -f "$SUPERVISOR_PIDFILE"; exit 0' INT TERM

echo "$(date -u +%Y-%m-%dT%H:%M:%SZ) supervisor pid $$" >>"$RESTART_LOG"
start_child

while true; do
  sleep 10
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
  for _ in $(seq 1 25); do
    health_ok && break
    sleep 0.5
  done
done
