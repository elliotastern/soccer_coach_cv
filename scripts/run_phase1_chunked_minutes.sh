#!/usr/bin/env bash
# Chunked first-N-minutes batch so MPS crashes don't lose the whole run.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
export PYTHONPATH="." PYTHONUNBUFFERED=1
export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/mpl-soccer}"
mkdir -p "$MPLCONFIGDIR"

VIDEO="${1:-data/raw/Match 3/P10-002.mp4}"
OUT="${2:-data/output/full_match_2min}"
TOTAL_FRAMES="${3:-7200}"   # 2 min @ 60fps
CHUNK="${4:-300}"
STEM="$(basename "$VIDEO" .mp4)"
LOGDIR="reports/eval_match3/improve_eng_loop"
mkdir -p "$LOGDIR" "$OUT"
LOG="$LOGDIR/phase1_${STEM}_chunked_${TOTAL_FRAMES}.log"
STATE="$LOGDIR/phase1_${STEM}_chunked.state"

start=0
if [[ -f "$STATE" ]]; then
  start="$(cat "$STATE")"
fi

echo "[$(date -u +%H:%M:%S)] chunked start=$start total=$TOTAL_FRAMES chunk=$CHUNK → $OUT/$STEM" | tee -a "$LOG"

while (( start < TOTAL_FRAMES )); do
  n=$(( TOTAL_FRAMES - start ))
  if (( n > CHUNK )); then n=$CHUNK; fi
  echo "[$(date -u +%H:%M:%S)] chunk start=$start n=$n" | tee -a "$LOG"
  ok=0
  for attempt in 1 2 3; do
    set +e
    python3 apps/batch_pipeline.py \
      --video "$VIDEO" \
      --output "$OUT" \
      --start-frame "$start" \
      --max-frames "$n" \
      >>"$LOG" 2>&1
    ec=$?
    set -e
    if (( ec == 0 )); then
      ok=1
      break
    fi
    echo "[$(date -u +%H:%M:%S)] chunk exit=$ec attempt=$attempt — sleep and retry" | tee -a "$LOG"
    sleep 15
  done
  if (( ok != 1 )); then
    echo "[$(date -u +%H:%M:%S)] SKIP chunk start=$start after retries" | tee -a "$LOG"
    start=$((start + n))
    echo "$start" > "$STATE"
    continue
  fi
  # Merge cumulative (best-effort; never abort the run)
  set +e
  python3 - <<PY
from pathlib import Path
import pandas as pd, json
run = Path("$OUT") / "$STEM"
cum = Path("$OUT") / "${STEM}_cumulative"
cum.mkdir(parents=True, exist_ok=True)
fd, ev = run / "frame_data.csv", run / "events.json"
cfd, cev = cum / "frame_data.csv", cum / "events.json"
if fd.exists():
    df = pd.read_csv(fd)
    if cfd.exists():
        old = pd.read_csv(cfd)
        df = pd.concat([old, df], ignore_index=True).drop_duplicates(
            subset=["frame_id", "Player_ID", "Event"], keep="last"
        )
    df.to_csv(cfd, index=False)
if ev.exists():
    cur = json.loads(ev.read_text())
    events = list(cur.get("events") or [])
    if cev.exists():
        prev = json.loads(cev.read_text())
        seen = {e.get("id") for e in (prev.get("events") or [])}
        merged = list(prev.get("events") or [])
        for e in events:
            if e.get("id") not in seen:
                merged.append(e)
        events = merged
    cev.write_text(json.dumps({
        "match_id": cur.get("match_id") or "$STEM",
        "events": events,
        "metadata": {"total_frames": $TOTAL_FRAMES, "chunks_merged": True},
    }, indent=2))
print("cumulative rows", len(pd.read_csv(cfd)) if cfd.exists() else 0,
      "events", len(json.loads(cev.read_text())["events"]) if cev.exists() else 0)
PY
  set -e
  start=$((start + n))
  echo "$start" > "$STATE"
  echo "[$(date -u +%H:%M:%S)] advanced to $start/$TOTAL_FRAMES" | tee -a "$LOG"
  sleep 8
done

echo "[$(date -u +%H:%M:%S)] CHUNKED_DONE" | tee -a "$LOG"
# promote cumulative to run dir for Streamlit
cp -f "$OUT/${STEM}_cumulative/frame_data.csv" "$OUT/$STEM/frame_data.csv" 2>/dev/null || true
cp -f "$OUT/${STEM}_cumulative/events.json" "$OUT/$STEM/events.json" 2>/dev/null || true
echo "DONE → $OUT/$STEM"
