#!/usr/bin/env bash
# Eng-loop: Frame review stability ≥9/10 (no Errno 5 / Frame review failed).
set -uo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
export PYTHONPATH="." MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/mpl-soccer}"
PY="${PYTHON:-/Users/elliotstern/.venvs/soccer-rfdetr312/bin/python3}"
PORT="${1:-8501}"
N="${2:-10}"
OUT="reports/eval_match3/improve_eng_loop/frame_review_eio"
LOG="$OUT/eng_loop.log"
SCORE="$OUT/stability_score.json"
mkdir -p "$OUT" "$MPLCONFIGDIR"
: > "$LOG"

kill_st() {
  pkill -f "streamlit run apps/review_dashboard" 2>/dev/null || true
  screen -S st_review -X quit 2>/dev/null || true
  sleep 1
}

start_st() {
  kill_st
  screen -dmS st_review bash -lc "cd '$ROOT' && PYTHONPATH=. MPLCONFIGDIR='$MPLCONFIGDIR' \
    '$PY' -m streamlit run apps/review_dashboard.py \
      --server.address 127.0.0.1 \
      --server.port $PORT \
      --server.headless true \
      --browser.gatherUsageStats false \
      --server.fileWatcherType none \
    > '$OUT/streamlit.log' 2>&1"
  for i in $(seq 1 45); do
    code=$(curl -s -o /dev/null -w '%{http_code}' --connect-timeout 1 "http://127.0.0.1:$PORT/" || echo 000)
    if [[ "$code" == "200" ]]; then
      echo "up code=$code after ${i}s" | tee -a "$LOG"
      return 0
    fi
    sleep 1
  done
  echo "failed to come up" | tee -a "$LOG"
  return 1
}

echo "frame-review eng-loop N=$N port=$PORT" | tee -a "$LOG"
start_st | tee -a "$LOG"

"$PY" - <<PY | tee -a "$LOG"
import json, time, traceback
from pathlib import Path
from streamlit.testing.v1 import AppTest

N = $N
OUT = Path("$OUT")
results = []
for i in range(N):
    t0 = time.time()
    row = {"i": i, "ok": False, "eio": False, "errors": [], "exceptions": [], "sec": 0.0}
    try:
        at = AppTest.from_file("apps/review_dashboard.py", default_timeout=120)
        # Ensure play stays OFF (play+dets hammers LaCie)
        at.session_state["verify_playing"] = False
        at.session_state["show_dets"] = False
        at.run()
        errs = [str(x) for x in at.error]
        excs = [str(x) for x in at.exception]
        blob = "\n".join(errs + excs)
        eio = ("Errno 5" in blob) or ("Input/output error" in blob) or ("Frame review failed" in blob)
        row.update({
            "ok": (not eio) and (not excs),
            "eio": eio,
            "errors": errs[:6],
            "exceptions": excs[:6],
            "sec": round(time.time() - t0, 2),
            "headers": len(at.header),
        })
    except Exception as exc:
        row["errors"] = [repr(exc)]
        row["eio"] = "Errno 5" in str(exc) or "Input/output" in str(exc)
        row["sec"] = round(time.time() - t0, 2)
        traceback.print_exc()
    results.append(row)
    print(f"[{i}] ok={row['ok']} eio={row['eio']} sec={row['sec']} errs={row['errors'][:1]}")

ok = sum(1 for r in results if r["ok"])
score = round(10.0 * ok / max(1, N), 1)
payload = {
    "score": score,
    "ok": ok,
    "fail": N - ok,
    "n": N,
    "gate": 9.0,
    "pass": score >= 9.0,
    "url": "http://127.0.0.1:$PORT",
    "results": results,
    "fixes": [
        "EIO retries on video/csv reads",
        "RF-DETR off by default; play no longer forces dets",
        "soft-fail mosaic/fuse/detect",
        "removed components.html scroll iframe",
        "pause play + log /tmp/scv_frame_review_errors.log on EIO",
    ],
}
OUT.joinpath("stability_score.json").write_text(json.dumps(payload, indent=2))
print(json.dumps({k: payload[k] for k in ("score", "ok", "fail", "n", "pass", "gate")}, indent=2))
print(f"FRAME_REVIEW_SCORE {score}/10 gate={'PASS' if score >= 9 else 'FAIL'}")
PY
