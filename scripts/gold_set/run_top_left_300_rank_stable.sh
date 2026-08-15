#!/bin/bash
set -euo pipefail
ROOT="/Volumes/LaCie/Projects/Soccer Coach CV"
export TMPDIR="$ROOT/.tmp"
mkdir -p "$TMPDIR"
OUT="$ROOT/reports/eval_match2_v10/top_left_300_postproc_rank"
mkdir -p "$OUT"
LOG="$OUT/run.log"
PY=/Users/elliotstern/.venvs/soccer-rfdetr312/bin/python3
cd "$ROOT"
# kill prior rank jobs
pkill -f 'eval_top_left_300_postproc_rank.py' 2>/dev/null || true
sleep 1
nohup "$PY" -u scripts/gold_set/eval_top_left_300_postproc_rank.py --resume >>"$LOG" 2>&1 &
echo $! > "$OUT/run.pid"
echo "pid=$(cat "$OUT/run.pid") log=$LOG"
