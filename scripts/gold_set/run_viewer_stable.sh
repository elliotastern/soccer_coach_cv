#!/usr/bin/env bash
# Durable annotation viewer: threaded serve_viewer + PID file + healthcheck.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
PORT="${1:-8765}"
LOG="${TMPDIR:-/tmp}/soccer_coach_serve_viewer.log"
PIDFILE="${TMPDIR:-/tmp}/soccer_coach_serve_viewer.pid"
PORTFILE="${TMPDIR:-/tmp}/soccer_coach_serve_viewer.port"
EDITOR="http://127.0.0.1:${PORT}/data/processed/gold_sets/match1_1_100/review/editor.html"
FRAME="http://127.0.0.1:${PORT}/data/processed/gold_sets/match1_1_100/review/frames/000.jpg"

is_healthy() {
  local p="$1"
  curl -fsS -m 2 -o /dev/null \
    "http://127.0.0.1:${p}/data/processed/gold_sets/match1_1_100/review/frames/000.jpg"
}

# Reuse healthy existing server
if [[ -f "$PORTFILE" ]]; then
  old_port="$(cat "$PORTFILE" 2>/dev/null || true)"
  if [[ -n "${old_port:-}" ]] && is_healthy "$old_port"; then
    echo "Viewer already healthy on ${old_port}"
    echo "Gold100: http://127.0.0.1:${old_port}/data/processed/gold_sets/match1_1_100/review/editor.html"
    if [[ "${OPEN_BROWSER:-1}" == "1" ]]; then
      open "http://127.0.0.1:${old_port}/data/processed/gold_sets/match1_1_100/review/editor.html" || true
    fi
    exit 0
  fi
fi

# Stop stale viewers
if [[ -f "$PIDFILE" ]]; then
  old_pid="$(cat "$PIDFILE" 2>/dev/null || true)"
  if [[ -n "${old_pid:-}" ]] && kill -0 "$old_pid" 2>/dev/null; then
    kill "$old_pid" 2>/dev/null || true
    sleep 0.5
    kill -9 "$old_pid" 2>/dev/null || true
  fi
fi
pkill -f "serve_viewer.py" 2>/dev/null || true
sleep 0.5

cd "$ROOT"
# Detach fully so Cursor shell teardown does not kill the server
nohup python3 -u serve_viewer.py --port "$PORT" >"$LOG" 2>&1 </dev/null &
disown || true

# Wait for health (up to ~15s)
ok=0
for i in $(seq 1 30); do
  if is_healthy "$PORT"; then
    ok=1
    break
  fi
  sleep 0.5
done

if [[ "$ok" != "1" ]]; then
  echo "Viewer failed to become healthy on ${PORT}"
  echo "---- log ----"
  tail -40 "$LOG" || true
  exit 1
fi

# Concurrent stress: black screen was caused by single-thread deadlock
python3 - <<PY
import concurrent.futures, urllib.request
base="http://127.0.0.1:${PORT}/data/processed/gold_sets/match1_1_100/review/frames"
urls=[f"{base}/{i:03d}.jpg" for i in range(12)]
def fetch(u):
    with urllib.request.urlopen(u, timeout=10) as r:
        return r.status, int(r.headers.get("Content-Length") or 0)
with concurrent.futures.ThreadPoolExecutor(max_workers=12) as ex:
    results=list(ex.map(fetch, urls))
bad=[r for r in results if r[0]!=200 or r[1]<1000]
print(f"concurrent frames: {len(results)-len(bad)}/{len(results)} ok")
if bad:
    raise SystemExit("concurrent fetch failed")
PY

echo "Viewer healthy on ${PORT}"
echo "Gold100: ${EDITOR}"
echo "Log: ${LOG}"
if [[ "${OPEN_BROWSER:-1}" == "1" ]]; then
  open "$EDITOR" || true
fi
