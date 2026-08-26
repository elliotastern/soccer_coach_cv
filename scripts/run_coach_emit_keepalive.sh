#!/usr/bin/env bash
set -uo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
export PYTHONPATH="."
export OBJC_DISABLE_INITIALIZE_FORK_SAFETY=YES
PORT="${PORT:-8502}"
LOG="${LOG:-/tmp/coach_emit_${PORT}.log}"
HEALTH="http://127.0.0.1:${PORT}/_stcore/health"
ST="/Library/Frameworks/Python.framework/Versions/3.13/bin/streamlit"
healthy() { curl -sf --max-time 2 "$HEALTH" >/dev/null 2>&1; }
echo "$(date -u +%Y-%m-%dT%H:%M:%SZ) keepalive monitor port=${PORT}" >> "${LOG}.keepalive"
while true; do
  if healthy; then sleep 4; continue; fi
  for p in $(lsof -nP -iTCP:"$PORT" -sTCP:LISTEN 2>/dev/null | awk 'NR>1{print $2}'); do
    kill "$p" 2>/dev/null || true
  done
  sleep 1
  echo "$(date -u +%Y-%m-%dT%H:%M:%SZ) down — starting" >> "${LOG}.keepalive"
  "$ST" run "$ROOT/apps/coach_emit_label_dashboard.py" \
    --server.headless true --server.address 127.0.0.1 \
    --server.port "$PORT" --browser.serverPort "$PORT" \
    --server.fileWatcherType none --browser.gatherUsageStats false \
    >> "$LOG" 2>&1
  echo "$(date -u +%Y-%m-%dT%H:%M:%SZ) exit=$? — retry in 4s" >> "${LOG}.keepalive"
  sleep 4
done
