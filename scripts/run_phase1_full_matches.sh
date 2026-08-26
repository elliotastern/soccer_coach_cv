#!/usr/bin/env bash
# Phase 1 acceptance: two full Match 3 cams through batch_pipeline (no max-frames).
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
export PYTHONPATH="."
export PYTHONUNBUFFERED=1
export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/mpl-soccer}"
mkdir -p "$MPLCONFIGDIR" reports/eval_match3/improve_eng_loop data/output/full_match

OUT="data/output/full_match"
LOGDIR="reports/eval_match3/improve_eng_loop"
TS="$(date -u +%Y%m%dT%H%M%SZ)"
MASTER="$LOGDIR/phase1_full_matches_${TS}.log"

run_one() {
  local video="$1"
  local name
  name="$(basename "$video" .mp4)"
  local log="$LOGDIR/phase1_full_${name}_${TS}.log"
  echo "[$(date -u +%H:%M:%S)] START $name → $OUT/$name" | tee -a "$MASTER"
  python3 apps/batch_pipeline.py \
    --video "$video" \
    --output "$OUT" \
    >"$log" 2>&1
  local ec=$?
  echo "[$(date -u +%H:%M:%S)] END $name exit=$ec (see $log)" | tee -a "$MASTER"
  echo "EXIT:$ec" >>"$log"
  return "$ec"
}

echo "Phase 1 full matches started $TS" | tee "$MASTER"
run_one "data/raw/Match 3/P10-002.mp4"
run_one "data/raw/Match 3/P1-006.mp4"
echo "[$(date -u +%H:%M:%S)] BOTH_DONE" | tee -a "$MASTER"
