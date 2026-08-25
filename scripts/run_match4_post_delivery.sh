#!/usr/bin/env bash
# After git pull: audit Match 4 batch + render 5min mosaic (Catch or Mac with videos).
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
export PYTHONPATH="." PYTHONUNBUFFERED=1

echo "=== Batch events audit ==="
python3 scripts/gold_set/audit_batch_events_pack.py

echo "=== Fuse eval window (start 1200) ==="
python3 scripts/gold_set/build_fuse_eval_window.py \
  --start 1200 --match-sec 15 --stride 4 \
  --clip-id real_fuse_eval_20s \
  --labels-json data/processed/gold_sets/match3_events_v2_dribble/clips/real_fuse_eval_20s/labels.json

echo "=== Event eng-loops ==="
python3 scripts/eng_loop_heuristic_events.py

OUT_DIR="reports/eval_match3/improve_eng_loop/match4_5min"
mkdir -p "$OUT_DIR"
echo "=== 5min mosaic render (stride 15, ~hours on Mac) ==="
python3 scripts/gold_set/render_phase1_check_mosaic.py \
  --start 0 --match-sec 300 --stride 15 --out-fps 4 \
  --out-dir "$OUT_DIR" \
  --out-file coach_mosaic_pitch_5min.mp4

echo "DONE → $OUT_DIR/coach_mosaic_pitch_5min.mp4"
