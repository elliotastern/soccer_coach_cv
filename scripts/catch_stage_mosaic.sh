#!/usr/bin/env bash
# Catch: poll until mosaic RENDER_DONE, then stage to soccer_exchange.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
SRC="$ROOT/reports/eval_match3/improve_eng_loop/match4_5min/coach_mosaic_pitch_5min.mp4"
DEST="$HOME/soccer_exchange/from_catch/coach_mosaic_pitch_5min.mp4"
LOG="$ROOT/reports/eval_match3/improve_eng_loop/match4_5min_rerender.log"
INTERVAL="${POLL_SEC:-30}"

mkdir -p "$(dirname "$DEST")"
echo "mosaic_watch started $(date -u +%FT%TZ)"
while true; do
  if grep -q RENDER_DONE "$LOG" 2>/dev/null && [[ -f "$SRC" ]]; then
    # Prefer finalized file after h264_encode (size stable)
    sz1=$(stat -c%s "$SRC" 2>/dev/null || echo 0)
    sleep 5
    sz2=$(stat -c%s "$SRC" 2>/dev/null || echo 0)
    if [[ "$sz1" == "$sz2" && "$sz1" -gt 1000000 ]]; then
      cp -f "$SRC" "$DEST"
      echo "staged $(date -u +%FT%TZ) size=$sz1 → $DEST"
      exit 0
    fi
  fi
  # Progress line if render still running
  if [[ -f "$LOG" ]]; then
    tail -1 "$LOG" || true
  fi
  sleep "$INTERVAL"
done
