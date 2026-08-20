#!/usr/bin/env bash
# Eng-loop: keep Streamlit review up until ≥9/10 HTTP health checks pass.
set -uo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
export PYTHONPATH="." MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/mpl-soccer}"
PORT="${1:-8501}"
N="${2:-10}"
LOG="reports/eval_match3/improve_eng_loop/streamlit_eng_loop.log"
SCORE="reports/eval_match3/improve_eng_loop/streamlit_stability_score.json"
mkdir -p reports/eval_match3/improve_eng_loop "$MPLCONFIGDIR"

# refresh snapshot + static HTML fallback
python3 scripts/build_review_partial_html.py >>"$LOG" 2>&1 || true

kill_st() {
  pkill -f "streamlit run apps/review_dashboard" 2>/dev/null || true
  screen -S st_review -X quit 2>/dev/null || true
  sleep 1
}

start_st() {
  kill_st
  screen -dmS st_review bash -lc "cd '$ROOT' && PYTHONPATH=. MPLCONFIGDIR='$MPLCONFIGDIR' \
    streamlit run apps/review_dashboard.py \
      --server.address 127.0.0.1 \
      --server.port $PORT \
      --server.headless true \
      --browser.gatherUsageStats false \
      --server.fileWatcherType none \
    > reports/eval_match3/improve_eng_loop/streamlit_review.log 2>&1"
  # wait for listen
  for i in $(seq 1 30); do
    code=$(curl -s -o /dev/null -w '%{http_code}' --connect-timeout 1 "http://127.0.0.1:$PORT/" || echo 000)
    if [[ "$code" == "200" ]]; then
      echo "up code=$code after ${i}s"
      return 0
    fi
    sleep 1
  done
  echo "failed to come up"
  return 1
}

: > "$LOG"
ok=0
echo "streamlit eng-loop N=$N port=$PORT" | tee -a "$LOG"
start_st | tee -a "$LOG"

for i in $(seq 1 "$N"); do
  code=$(curl -s -o /dev/null -w '%{http_code}' --connect-timeout 2 "http://127.0.0.1:$PORT/" || echo 000)
  # also hit healthz-ish by fetching page content length
  bytes=$(curl -s --connect-timeout 2 "http://127.0.0.1:$PORT/" | wc -c | tr -d ' ')
  if [[ "$code" == "200" && "$bytes" -gt 500 ]]; then
    ok=$((ok + 1))
    echo "[$(date -u +%H:%M:%S)] PASS $i/$N http=$code bytes=$bytes" | tee -a "$LOG"
  else
    echo "[$(date -u +%H:%M:%S)] FAIL $i/$N http=$code bytes=$bytes — restart" | tee -a "$LOG"
    start_st | tee -a "$LOG" || true
    sleep 2
  fi
  sleep 2
done

python3 - <<PY
import json
from pathlib import Path
ok, n = $ok, $N
score = round(10.0 * ok / max(1, n), 1)
payload = {
    "score": score,
    "ok": ok,
    "fail": n - ok,
    "n": n,
    "gate": 9.0,
    "pass": score >= 9.0,
    "url": "http://127.0.0.1:$PORT",
    "static_html": "reports/eval_match3/improve_eng_loop/review_partial.html",
    "output_root": "data/output/full_match_2min_partial",
}
Path("$SCORE").write_text(json.dumps(payload, indent=2))
print(json.dumps(payload, indent=2))
print(f"STREAMLIT_SCORE {score}/10 gate={'PASS' if score >= 9 else 'FAIL'}")
PY
