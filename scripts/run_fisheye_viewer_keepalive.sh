#!/usr/bin/env bash
# Keep Match 3 fisheye dashboard viewer up on 8081 (serve_viewer.py).
# Handover often owns 8080 without fisheye routes — use 8081.
#
# Critical: detach the Python child with nohup+disown. Cursor agent shells
# SIGTERM/SIGKILL process groups when a turn ends; a foreground serve_viewer
# (or a child still in that session) dies and the browser shows Connection Failed.
set -uo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
PORT="${PORT:-8081}"
LOG="${LOG:-/tmp/serve_viewer_${PORT}.log}"
PIDFILE="${PIDFILE:-/tmp/serve_viewer_${PORT}.pid}"
KA_PIDFILE="${KA_PIDFILE:-/tmp/serve_viewer_${PORT}.keepalive.pid}"
HEALTH="http://127.0.0.1:${PORT}/reports/eval_match3/fisheye_dashboard/index.html"
PY="${PYTHON:-python3}"

echo $$ >"$KA_PIDFILE"
healthy() { curl -sf --max-time 3 -o /dev/null "$HEALTH"; }

listener_pids() {
  lsof -nP -iTCP:"$PORT" -sTCP:LISTEN 2>/dev/null | awk 'NR>1{print $2}' | sort -u
}

stop_port() {
  local p
  for p in $(listener_pids); do
    kill "$p" 2>/dev/null || true
  done
  sleep 0.5
  for p in $(listener_pids); do
    kill -9 "$p" 2>/dev/null || true
  done
}

start_viewer() {
  stop_port
  sleep 0.3
  echo "$(date -u +%Y-%m-%dT%H:%M:%SZ) down — starting serve_viewer" >>"${LOG}.keepalive"
  # Detach from this shell's session so Cursor teardown cannot kill it.
  nohup "$PY" -u "$ROOT/serve_viewer.py" --port "$PORT" --no-fallback \
    >>"$LOG" 2>&1 </dev/null &
  local child=$!
  echo "$child" >"$PIDFILE"
  disown "$child" 2>/dev/null || true
  local i
  for i in $(seq 1 30); do
    if healthy; then
      echo "$(date -u +%Y-%m-%dT%H:%M:%SZ) up pid=${child}" >>"${LOG}.keepalive"
      return 0
    fi
    if ! kill -0 "$child" 2>/dev/null; then
      echo "$(date -u +%Y-%m-%dT%H:%M:%SZ) child exited before healthy" >>"${LOG}.keepalive"
      return 1
    fi
    sleep 0.4
  done
  echo "$(date -u +%Y-%m-%dT%H:%M:%SZ) start timeout" >>"${LOG}.keepalive"
  return 1
}

echo "$(date -u +%Y-%m-%dT%H:%M:%SZ) fisheye viewer keepalive port=${PORT} pid=$$" >>"${LOG}.keepalive"
while true; do
  if healthy; then
    sleep 4
    continue
  fi
  start_viewer || true
  sleep 3
done
