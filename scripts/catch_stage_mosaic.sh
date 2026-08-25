#!/usr/bin/env bash
# Catch: when mosaic render finishes, copy to soccer_exchange for Mac pull.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
SRC="$ROOT/reports/eval_match3/improve_eng_loop/match4_5min/coach_mosaic_pitch_5min.mp4"
DEST="$HOME/soccer_exchange/from_catch/coach_mosaic_pitch_5min.mp4"
LOG="$ROOT/reports/eval_match3/improve_eng_loop/match4_5min_rerender.log"
if grep -q RENDER_DONE "$LOG" 2>/dev/null && [[ -f "$SRC" ]]; then
  mkdir -p "$(dirname "$DEST")"
  cp -f "$SRC" "$DEST"
  echo "staged $(date -u +%FT%TZ) → $DEST"
fi
