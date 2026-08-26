#!/usr/bin/env bash
# Match 3 (Catch raw folder) — full quad batch + 3-min mosaic/best-ball MP4 checkpoints.
# Uses v12_hard ball + people checkpoints via configs/default.yaml merge.
set -uo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
export PYTHONPATH="." PYTHONUNBUFFERED=1
export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/mpl-soccer}"
mkdir -p "$MPLCONFIGDIR" reports/eval_match3/improve_eng_loop

OUT="${1:-data/output/match_3_full}"
TOTAL="${2:-96827}"
SEG_FRAMES="${3:-10800}"   # 3 min @ 60 fps
INNER_CHUNK="${4:-1800}" # 30 s batch sub-chunks
RAW="data/raw/Match 3"
LOGDIR="reports/eval_match3/improve_eng_loop"
RENDIR="reports/eval_match3/improve_eng_loop/match3_full_delivery"
EXCHANGE="${HOME}/soccer_exchange/from_catch/match3_full"
TS="$(date -u +%Y%m%dT%H%M%SZ)"
MASTER="$LOGDIR/match3_full_3min_${TS}.log"
CFG="$LOGDIR/_batch_rtx5090_merged.yaml"
STATE="$LOGDIR/match3_full_3min.state"
CAMS="${CAMS:-P10-match4 P9-match4 P7-match4 P8-match4}"
SRC_FPS="${SRC_FPS:-60}"
STRIDE="${STRIDE:-4}"
OUT_FPS="${OUT_FPS:-15}"

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
# Product ball model (v12 hard) — explicit for delivery runs.
base.setdefault("detection", {})["ball_checkpoint"] = (
    "models/v12_hard_snaps/post_train/checkpoint.pth"
)
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
}

run_cam_segment() {
  local stem="$1" seg_start="$2" seg_n="$3"
  local video="$RAW/${stem}.mp4"
  local log="$LOGDIR/batch_${stem}_full_${TS}.log"
  local cum="$OUT/${stem}_cumulative"
  [[ -f "$video" ]] || { echo "SKIP $video" | tee -a "$MASTER"; return 0; }
  mkdir -p "$OUT" "$cum" "$OUT/$stem"
  local pos=0
  while (( pos < seg_n )); do
    local n=$((seg_n - pos))
    (( n > INNER_CHUNK )) && n=$INNER_CHUNK
    local abs=$((seg_start + pos))
    echo "[$(date -u +%H:%M:%S)] $stem seg@${seg_start} +$pos n=$n (abs=$abs)" | tee -a "$MASTER" "$log"
    if ! python3 apps/batch_pipeline.py \
      --video "$video" --config "$CFG" --output "$OUT" \
      --start-frame "$abs" --max-frames "$n" >>"$log" 2>&1; then
      echo "FAIL $stem at $abs" | tee -a "$MASTER"
      return 1
    fi
    merge_promote "$stem" "$cum" "$OUT/$stem" "$TOTAL"
    pos=$((pos + n))
  done
}

render_segment() {
  local seg_idx="$1" seg_start="$2" seg_n="$3"
  local seg_sec
  seg_sec="$(python3 - <<PY
print(round($seg_n / $SRC_FPS, 3))
PY
)"
  local tag
  tag="$(printf 'seg%03d_fr%06d' "$seg_idx" "$seg_start")"
  local seg_out="$RENDIR/$tag"
  mkdir -p "$seg_out" "$EXCHANGE"
  echo "[$(date -u +%H:%M:%S)] RENDER $tag ${seg_sec}s start=$seg_start" | tee -a "$MASTER"
  for layout in mosaic best_ball; do
    local fname="coach_${layout}_${tag}.mp4"
    if ! python3 scripts/gold_set/render_phase1_check_mosaic.py \
      --start "$seg_start" \
      --match-sec "$seg_sec" \
      --src-fps "$SRC_FPS" \
      --stride "$STRIDE" \
      --out-fps "$OUT_FPS" \
      --layout "$layout" \
      --out-dir "$seg_out" \
      --out-file "$fname" >>"$MASTER" 2>&1; then
      echo "RENDER_FAIL $fname" | tee -a "$MASTER"
      return 1
    fi
    cp -f "$seg_out/$fname" "$EXCHANGE/$fname"
  done
  echo "STAGED $tag → $EXCHANGE" | tee -a "$MASTER"
}

seg_start=0
seg_idx=0
if [[ -f "$STATE" ]]; then
  read -r seg_start seg_idx <"$STATE" || true
fi

echo "Match 3 FULL 3-min delivery $TS → $OUT total=$TOTAL seg=$SEG_FRAMES" | tee "$MASTER"
mkdir -p "$RENDIR" "$EXCHANGE"

while (( seg_start < TOTAL )); do
  seg_n=$((TOTAL - seg_start))
  (( seg_n > SEG_FRAMES )) && seg_n=$SEG_FRAMES
  echo "[$(date -u +%H:%M:%S)] === SEG $seg_idx start=$seg_start n=$seg_n ===" | tee -a "$MASTER"
  for c in $CAMS; do
    run_cam_segment "$c" "$seg_start" "$seg_n" || exit 1
  done
  render_segment "$seg_idx" "$seg_start" "$seg_n" || exit 1
  seg_start=$((seg_start + seg_n))
  seg_idx=$((seg_idx + 1))
  echo "$seg_start $seg_idx" >"$STATE"
done

echo "ALL_DONE → $OUT renders → $EXCHANGE" | tee -a "$MASTER"
