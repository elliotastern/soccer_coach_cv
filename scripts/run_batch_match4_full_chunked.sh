#!/usr/bin/env bash
# Match 4 — full match chunked (quad default). Resume via state files.
set -uo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
export PYTHONPATH="." PYTHONUNBUFFERED=1
export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/mpl-soccer}"
mkdir -p "$MPLCONFIGDIR" reports/eval_match3/improve_eng_loop

OUT="${1:-data/output/match_4_full}"
TOTAL="${2:-96827}"
CHUNK="${3:-1800}"
RAW="data/raw/Match 3"
LOGDIR="reports/eval_match3/improve_eng_loop"
TS="$(date -u +%Y%m%dT%H%M%SZ)"
MASTER="$LOGDIR/batch_match4_full_${TS}.log"
CFG="$LOGDIR/_batch_rtx5090_merged.yaml"
CAMS="${CAMS:-P10-match4 P9-match4 P7-match4 P8-match4}"

# Kit-ref: env → match root → P10 cam
if [[ -z "${KIT_REF:-}" ]]; then
  if [[ -f "$OUT/team_centroids.json" ]]; then
    KIT_REF="$OUT/team_centroids.json"
  elif [[ -f "$OUT/P10-match4/team_centroids.json" ]]; then
    KIT_REF="$OUT/P10-match4/team_centroids.json"
  fi
fi
if [[ -n "${KIT_REF:-}" && -f "$KIT_REF" ]]; then
  echo "KIT_REF=$KIT_REF — seeding all cams under $OUT"
  for c in $CAMS; do
    mkdir -p "$OUT/$c"
    cp -f "$KIT_REF" "$OUT/$c/team_centroids.json"
  done
  mkdir -p "$OUT"
  cp -f "$KIT_REF" "$OUT/team_centroids.json"
else
  echo "WARN: no KIT_REF / team_centroids.json — cams will Golden Batch cluster"
fi

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
        "metadata": {"total_frames": total, "full_match": True},
    }, indent=2))
n = len(pd.read_csv(cfd)["frame_id"].unique()) if cfd.exists() else 0
print(f"promoted {stem}: {n} frames")
PY
  cp -f "$cum/frame_data.csv" "$OUT/$stem/" 2>/dev/null || true
  cp -f "$cum/events.json" "$OUT/$stem/" 2>/dev/null || true
  cp -f "$cum/events.csv" "$OUT/$stem/" 2>/dev/null || true
  if [[ -n "${KIT_REF:-}" && -f "$KIT_REF" ]]; then
    cp -f "$KIT_REF" "$OUT/$stem/team_centroids.json"
  elif [[ -f "$cum/team_centroids.json" ]]; then
    cp -f "$cum/team_centroids.json" "$OUT/$stem/team_centroids.json"
  fi
}

run_cam() {
  local stem="$1"
  local video="$RAW/${stem}.mp4"
  local log="$LOGDIR/batch_${stem}_full_${TS}.log"
  local state="$LOGDIR/phase1_${stem}_full.state"
  local cum="$OUT/${stem}_cumulative"
  [[ -f "$video" ]] || { echo "SKIP $video" | tee -a "$MASTER"; return 0; }
  local start=0
  [[ -f "$state" ]] && start="$(cat "$state")"
  mkdir -p "$OUT" "$cum" "$OUT/$stem"
  if [[ -n "${KIT_REF:-}" && -f "$KIT_REF" ]]; then
    cp -f "$KIT_REF" "$OUT/$stem/team_centroids.json"
  fi
  while (( start < TOTAL )); do
    local n=$((TOTAL - start)); (( n > CHUNK )) && n=$CHUNK
    echo "[$(date -u +%H:%M:%S)] $stem $start/$TOTAL n=$n" | tee -a "$MASTER" "$log"
    if ! python3 apps/batch_pipeline.py \
      --video "$video" --config "$CFG" --output "$OUT" \
      --start-frame "$start" --max-frames "$n" >>"$log" 2>&1; then
      echo "FAIL $stem at $start" | tee -a "$MASTER"
      return 1
    fi
    merge_promote "$stem" "$cum" "$OUT/$stem" "$TOTAL"
    start=$((start + n))
    echo "$start" > "$state"
  done
  echo "[$(date -u +%H:%M:%S)] DONE $stem" | tee -a "$MASTER"
}

echo "Match 4 FULL batch $TS → $OUT total=$TOTAL" | tee "$MASTER"
for c in $CAMS; do run_cam "$c"; done
echo "ALL_DONE → $OUT" | tee -a "$MASTER"
