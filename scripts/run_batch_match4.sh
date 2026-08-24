#!/usr/bin/env bash
# Batch Match 4 symlinks (P*-match4.mp4) — locked cam mapping, see match4_camera_ids.mdc.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
export PYTHONPATH="."
export PYTHONUNBUFFERED=1
export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/mpl-soccer}"
mkdir -p "$MPLCONFIGDIR"

OUT="${1:-data/output/match_4}"
RAW="data/raw/Match 3"
LOGDIR="reports/eval_match3/improve_eng_loop"
TS="$(date -u +%Y%m%dT%H%M%SZ)"
MASTER="$LOGDIR/batch_match4_${TS}.log"
mkdir -p "$LOGDIR" "$OUT"

# Quad first (mosaic), then ends, then goals.
CAMS=(
  P10-match4 P9-match4 P7-match4 P8-match4
  P1-match4 P6-match4
  P_Goal1-match4 P_Goal2-match4
)

run_one() {
  local stem="$1"
  local video="$RAW/${stem}.mp4"
  local log="$LOGDIR/batch_${stem}_${TS}.log"
  [[ -f "$video" ]] || { echo "SKIP missing $video" | tee -a "$MASTER"; return 0; }
  echo "[$(date -u +%H:%M:%S)] START $stem → $OUT" | tee -a "$MASTER"
  python3 apps/batch_pipeline.py --video "$video" --output "$OUT" >"$log" 2>&1
  local ec=$?
  echo "[$(date -u +%H:%M:%S)] END $stem exit=$ec (see $log)" | tee -a "$MASTER"
  echo "EXIT:$ec" >>"$log"
  return "$ec"
}

echo "Match 4 batch started $TS → $OUT" | tee "$MASTER"
for c in "${CAMS[@]}"; do
  run_one "$c"
done
echo "[$(date -u +%H:%M:%S)] ALL_DONE" | tee -a "$MASTER"
echo "Review: Expert mode → output root $OUT"
