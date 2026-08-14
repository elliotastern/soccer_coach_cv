#!/usr/bin/env bash
# Durable annotation viewer: threaded serve_viewer + PID file + healthcheck.
# Prefer this over bare `python3 serve_viewer.py` in Cursor shells (those get SIGKILL'd).
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
PORT="${1:-8080}"
LOG="${TMPDIR:-/tmp}/soccer_coach_serve_viewer.log"
PIDFILE="${TMPDIR:-/tmp}/soccer_coach_serve_viewer.pid"
PORTFILE="${TMPDIR:-/tmp}/soccer_coach_serve_viewer.port"
LOCKFILE="${TMPDIR:-/tmp}/soccer_coach_serve_viewer.lock"
EDITOR="http://127.0.0.1:${PORT}/data/processed/gold_sets/match1_1_100/review/editor.html"
DASHBOARD="${OPEN_URL:-http://127.0.0.1:${PORT}/4quad}"

is_healthy() {
  local p="$1"
  curl -fsS -m 3 -o /dev/null \
    "http://127.0.0.1:${p}/data/processed/gold_sets/match1_1_100/review/frames/000.jpg" || return 1
  local code
  code="$(curl -s -m 3 -o /dev/null -w '%{http_code}' "http://127.0.0.1:${p}/4quad" || true)"
  [[ "$code" == "302" || "$code" == "200" ]]
}

stop_listener_on_port() {
  local p="$1"
  local pids
  pids="$(lsof -tiTCP:"$p" -sTCP:LISTEN 2>/dev/null || true)"
  if [[ -n "${pids:-}" ]]; then
    # shellcheck disable=SC2086
    kill $pids 2>/dev/null || true
    sleep 0.4
    # shellcheck disable=SC2086
    kill -9 $pids 2>/dev/null || true
  fi
}

# Serialize starts so overlapping watchdogs cannot thrash (macOS: no flock)
acquire_lock() {
  local waited=0
  while ! mkdir "$LOCKFILE" 2>/dev/null; do
    sleep 0.2
    waited=$((waited + 1))
    if [[ $waited -gt 100 ]]; then
      echo "Lock stuck at $LOCKFILE — removing stale lock"
      rmdir "$LOCKFILE" 2>/dev/null || true
    fi
  done
  trap 'rmdir "$LOCKFILE" 2>/dev/null || true' EXIT
}
acquire_lock

# Reuse healthy existing server
if is_healthy "$PORT"; then
  echo "Viewer already healthy on ${PORT}"
  echo "$PORT" >"$PORTFILE"
  lsof -tiTCP:"$PORT" -sTCP:LISTEN 2>/dev/null | head -1 >"$PIDFILE" || true
  echo "Dashboard: ${DASHBOARD}"
  if [[ "${OPEN_BROWSER:-1}" == "1" ]]; then
    open "$DASHBOARD" || true
  fi
  exit 0
fi

# Stop only the PID we own / whatever holds the port (never broad pkill -f)
if [[ -f "$PIDFILE" ]]; then
  old_pid="$(cat "$PIDFILE" 2>/dev/null || true)"
  if [[ -n "${old_pid:-}" ]] && kill -0 "$old_pid" 2>/dev/null; then
    kill "$old_pid" 2>/dev/null || true
    sleep 0.4
    kill -9 "$old_pid" 2>/dev/null || true
  fi
fi
stop_listener_on_port "$PORT"
sleep 0.3

cd "$ROOT"
# Detach fully so Cursor shell teardown does not kill the server.
# --no-fallback keeps dashboards on the requested port (never silent 8081).
nohup python3 -u serve_viewer.py --port "$PORT" --no-fallback >"$LOG" 2>&1 </dev/null &
new_pid=$!
echo "$new_pid" >"$PIDFILE"
echo "$PORT" >"$PORTFILE"
disown "$new_pid" 2>/dev/null || true

ok=0
for _ in $(seq 1 40); do
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

live_pid="$(lsof -tiTCP:"$PORT" -sTCP:LISTEN 2>/dev/null | head -1 || true)"
if [[ -n "${live_pid:-}" ]]; then
  echo "$live_pid" >"$PIDFILE"
fi

echo "Viewer healthy on ${PORT} pid=$(cat "$PIDFILE")"
echo "Dashboard: ${DASHBOARD}"
echo "4quad-cvat: http://127.0.0.1:${PORT}/4quad-cvat"
echo "Log: ${LOG}"
if [[ "${OPEN_BROWSER:-1}" == "1" ]]; then
  open "$DASHBOARD" || true
fi
