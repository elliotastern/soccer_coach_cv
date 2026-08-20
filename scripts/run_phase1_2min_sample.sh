#!/usr/bin/env bash
# 2-minute sample (7200 @ 60fps) using Mac-stable config + validated 150-frame chunks.
set -uo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
export PYTHONPATH="." PYTHONUNBUFFERED=1
export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/mpl-soccer}"
mkdir -p "$MPLCONFIGDIR" reports/eval_match3/improve_eng_loop

VIDEO="${1:-data/raw/Match 3/P10-002.mp4}"
OUT="${2:-data/output/full_match_2min}"
TOTAL="${3:-7200}"
CHUNK="${4:-150}"
STEM="$(basename "$VIDEO" .mp4)"
LOG="reports/eval_match3/improve_eng_loop/phase1_${STEM}_2min_sample.log"
STATE="reports/eval_match3/improve_eng_loop/phase1_${STEM}_2min_sample.state"
CFG="reports/eval_match3/improve_eng_loop/_batch_mac_stable_merged.yaml"
CUM="$OUT/${STEM}_cumulative"

python3 - <<'PY'
import yaml
from pathlib import Path
base = yaml.safe_load(Path("configs/default.yaml").read_text())
over = yaml.safe_load(Path("configs/batch_mac_stable.yaml").read_text())
for k, v in over.items():
    if isinstance(v, dict) and isinstance(base.get(k), dict):
        base[k].update(v)
    else:
        base[k] = v
Path("reports/eval_match3/improve_eng_loop/_batch_mac_stable_merged.yaml").write_text(yaml.dump(base))
PY

start=0
[[ -f "$STATE" ]] && start="$(cat "$STATE")"
mkdir -p "$OUT" "$CUM"
: >> "$LOG"
echo "[$(date -u +%H:%M:%S)] SAMPLE start=$start total=$TOTAL chunk=$CHUNK" | tee -a "$LOG"

while (( start < TOTAL )); do
  n=$((TOTAL - start))
  (( n > CHUNK )) && n=$CHUNK
  echo "[$(date -u +%H:%M:%S)] chunk start=$start n=$n" | tee -a "$LOG"
  ok=0
  for attempt in 1 2 3; do
    set +e
    python3 apps/batch_pipeline.py \
      --video "$VIDEO" --config "$CFG" --output "$OUT" \
      --start-frame "$start" --max-frames "$n" >>"$LOG" 2>&1
    ec=$?
    set -e
    if (( ec == 0 )) && tail -20 "$LOG" | grep -q "Processing complete!"; then
      ok=1
      break
    fi
    echo "[$(date -u +%H:%M:%S)] retry attempt=$attempt exit=$ec" | tee -a "$LOG"
    sleep 12
  done
  if (( ok != 1 )); then
    echo "[$(date -u +%H:%M:%S)] SKIP start=$start" | tee -a "$LOG"
    start=$((start + n))
    echo "$start" > "$STATE"
    continue
  fi
  set +e
  python3 - <<PY
from pathlib import Path
import pandas as pd, json
run = Path("$OUT") / "$STEM"
cum = Path("$CUM")
cum.mkdir(parents=True, exist_ok=True)
fd, ev = run / "frame_data.csv", run / "events.json"
cfd, cev = cum / "frame_data.csv", cum / "events.json"
if fd.exists():
    df = pd.read_csv(fd)
    if cfd.exists():
        df = pd.concat([pd.read_csv(cfd), df], ignore_index=True).drop_duplicates(
            subset=["frame_id", "Player_ID", "Event"], keep="last"
        )
    df.sort_values("frame_id").to_csv(cfd, index=False)
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
        "metadata": {"total_frames": $TOTAL, "sample_2min": True},
    }, indent=2))
    # also write events.csv for Phase 1 schema
    rows = []
    for e in events:
        loc = e.get("start_location") or {}
        rows.append({
            "Timestamp": e.get("timestamp_start", 0),
            "Team_ID": -1,
            "Player_ID": (e.get("involved_players") or [-1])[0],
            "Event": e.get("type", ""),
            "Location_X": loc.get("x", 0),
            "Location_Y": loc.get("y", 0),
            "frame_id": e.get("start_frame", 0),
            "confidence": e.get("confidence", 0),
        })
    if rows:
        pd.DataFrame(rows).to_csv(cum / "events.csv", index=False)
print("cum frames", len(pd.read_csv(cfd)["frame_id"].unique()) if cfd.exists() else 0)
PY
  set -e
  start=$((start + n))
  echo "$start" > "$STATE"
  echo "[$(date -u +%H:%M:%S)] advanced $start/$TOTAL" | tee -a "$LOG"
  sleep 2
done

# promote cumulative into run dir for Streamlit
cp -f "$CUM/frame_data.csv" "$OUT/$STEM/frame_data.csv" 2>/dev/null || true
cp -f "$CUM/events.json" "$OUT/$STEM/events.json" 2>/dev/null || true
cp -f "$CUM/events.csv" "$OUT/$STEM/events.csv" 2>/dev/null || true
echo "[$(date -u +%H:%M:%S)] SAMPLE_DONE → $OUT/$STEM" | tee -a "$LOG"
