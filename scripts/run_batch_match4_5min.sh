#!/usr/bin/env bash
# Match 4 — 5 min @ 60fps per cam, chunked for live Streamlit review (RTX 5090).
# Quad cams first (mosaic). Promotes cumulative CSV/JSON after every chunk.
set -uo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
export PYTHONPATH="." PYTHONUNBUFFERED=1
export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/mpl-soccer}"
mkdir -p "$MPLCONFIGDIR" reports/eval_match3/improve_eng_loop

OUT="${1:-data/output/match_4_5min}"
TOTAL="${2:-18000}"   # 5 min @ 60 fps
CHUNK="${3:-1800}"    # 30 s video per chunk (~4 min wall @ 8 fps)
RAW="data/raw/Match 3"
LOGDIR="reports/eval_match3/improve_eng_loop"
TS="$(date -u +%Y%m%dT%H%M%SZ)"
MASTER="$LOGDIR/batch_match4_5min_${TS}.log"
CFG="$LOGDIR/_batch_rtx5090_merged.yaml"

# Quad only by default — enough for coach mosaic. Override: CAMS="P10-match4" ...
CAMS="${CAMS:-P10-match4 P9-match4 P7-match4 P8-match4}"

python3 - <<'PY'
import yaml
from pathlib import Path

base = yaml.safe_load(Path("configs/default.yaml").read_text())
over = yaml.safe_load(Path("configs/batch_rtx5090.yaml").read_text())
for k, v in over.items():
    if isinstance(v, dict) and isinstance(base.get(k), dict):
        base[k].update(v)
    else:
        base[k] = v
Path("reports/eval_match3/improve_eng_loop/_batch_rtx5090_merged.yaml").write_text(
    yaml.dump(base)
)
PY

merge_promote() {
  local stem="$1" cum="$2" run="$3" total="$4"
  python3 - <<PY
from pathlib import Path
import json
import pandas as pd

stem, cum, run, total = "$stem", Path("$cum"), Path("$run"), int("$total")
run.mkdir(parents=True, exist_ok=True)
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
        "match_id": cur.get("match_id") or stem,
        "events": events,
        "metadata": {"total_frames": total, "sample_5min": True},
    }, indent=2))
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
n = len(pd.read_csv(cfd)["frame_id"].unique()) if cfd.exists() else 0
print(f"promoted {stem}: {n} frames → {run}")
PY
  cp -f "$cum/frame_data.csv" "$OUT/$stem/frame_data.csv" 2>/dev/null || true
  cp -f "$cum/events.json" "$OUT/$stem/events.json" 2>/dev/null || true
  cp -f "$cum/events.csv" "$OUT/$stem/events.csv" 2>/dev/null || true
}

run_cam() {
  local stem="$1"
  local video="$RAW/${stem}.mp4"
  local log="$LOGDIR/batch_${stem}_5min_${TS}.log"
  local state="$LOGDIR/phase1_${stem}_5min.state"
  local cum="$OUT/${stem}_cumulative"

  [[ -f "$video" ]] || { echo "SKIP missing $video" | tee -a "$MASTER"; return 0; }

  local start=0
  [[ -f "$state" ]] && start="$(cat "$state")"
  mkdir -p "$OUT" "$cum" "$OUT/$stem"
  : >> "$log"
  echo "[$(date -u +%H:%M:%S)] START $stem start=$start total=$TOTAL chunk=$CHUNK" \
    | tee -a "$MASTER" "$log"

  while (( start < TOTAL )); do
    local n=$((TOTAL - start))
    (( n > CHUNK )) && n=$CHUNK
    echo "[$(date -u +%H:%M:%S)] $stem chunk start=$start n=$n" | tee -a "$MASTER" "$log"
    local ok=0
    for attempt in 1 2 3; do
      set +e
      python3 apps/batch_pipeline.py \
        --video "$video" --config "$CFG" --output "$OUT" \
        --start-frame "$start" --max-frames "$n" >>"$log" 2>&1
      local ec=$?
      set -e
      if (( ec == 0 )) && tail -20 "$log" | grep -q "Processing complete!"; then
        ok=1
        break
      fi
      echo "[$(date -u +%H:%M:%S)] retry attempt=$attempt exit=$ec" | tee -a "$MASTER" "$log"
      sleep 8
    done
    if (( ok != 1 )); then
      echo "[$(date -u +%H:%M:%S)] SKIP $stem start=$start" | tee -a "$MASTER" "$log"
      start=$((start + n))
      echo "$start" > "$state"
      continue
    fi
    merge_promote "$stem" "$cum" "$OUT/$stem" "$TOTAL"
    start=$((start + n))
    echo "$start" > "$state"
    echo "[$(date -u +%H:%M:%S)] $stem advanced $start/$TOTAL — review Expert → $OUT" \
      | tee -a "$MASTER" "$log"
    sleep 1
  done
  echo "[$(date -u +%H:%M:%S)] DONE $stem" | tee -a "$MASTER" "$log"
}

echo "Match 4 5-min batch started $TS → $OUT (total=$TOTAL chunk=$CHUNK)" | tee "$MASTER"
echo "Config: $CFG (batch_rtx5090)" | tee -a "$MASTER"
echo "Review while running: Expert mode → output root $OUT" | tee -a "$MASTER"

for c in $CAMS; do
  run_cam "$c"
done

echo "[$(date -u +%H:%M:%S)] ALL_DONE → $OUT" | tee -a "$MASTER"
echo "Dashboard: http://127.0.0.1:8501 Expert mode → $OUT"
