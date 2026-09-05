#!/usr/bin/env bash
# Durable Match3 M1 / gold viewer on :8877 (double-fork; survives Cursor shells).
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
PY="${SERVE_VIEWER_PY:-/Library/Frameworks/Python.framework/Versions/3.13/bin/python3}"
PORT=8877
LOG=/tmp/serve_viewer_8877.out
PIDFILE=/tmp/serve_viewer_8877.pid

if lsof -nP -iTCP:${PORT} -sTCP:LISTEN >/dev/null 2>&1; then
  echo "already listening on ${PORT}"
  curl -sf -o /dev/null "http://127.0.0.1:${PORT}/match3-m1" && echo "ok http://127.0.0.1:${PORT}/match3-m1-blur"
  exit 0
fi

export ROOT PY PORT LOG PIDFILE
"$PY" - <<'PY'
import os, sys, time, socket
from pathlib import Path
root = Path(os.environ["ROOT"])
log = Path(os.environ["LOG"])
pidfile = Path(os.environ["PIDFILE"])
py = os.environ["PY"]
port = int(os.environ["PORT"])

s = socket.socket(); s.settimeout(0.3)
try:
    s.connect(("127.0.0.1", port)); raise SystemExit(0)
except OSError:
    pass
finally:
    s.close()

if os.fork() > 0:
    time.sleep(1.5); raise SystemExit(0)
os.setsid()
if os.fork() > 0:
    raise SystemExit(0)
os.chdir(root)
os.environ["PYTHONUNBUFFERED"] = "1"
log_f = open(log, "a", buffering=1)
os.dup2(log_f.fileno(), 1); os.dup2(log_f.fileno(), 2)
pidfile.write_text(str(os.getpid()))
os.execv(py, [py, "-u", str(root / "serve_viewer.py"), "--port", str(port)])
PY

sleep 1
lsof -nP -iTCP:${PORT} -sTCP:LISTEN
echo "http://127.0.0.1:${PORT}/match3-m1-blur"
