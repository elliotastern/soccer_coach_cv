#!/usr/bin/env bash
set -uo pipefail
cd "/Volumes/LaCie/Projects/Soccer Coach CV"
export PYTORCH_ENABLE_MPS_FALLBACK=1
export PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.0
LOG=reports/batch3_build.log
SCORE=data/processed/gold_sets/math_1_training_batch3/candidate_scores.json
echo "$(date +%Y-%m-%dT%H:%M:%S) SUPERVISOR start" | tee -a "$LOG"
score_n() {
  if [[ -f "$SCORE" ]]; then
    python3 -c "import json;print(len(json.load(open('$SCORE'))))"
  else
    echo 0
  fi
}
for attempt in $(seq 1 120); do
  if [[ -f data/processed/gold_sets/math_1_training_batch3/manifest.json ]]; then
    echo "$(date +%Y-%m-%dT%H:%M:%S) DONE manifest exists" | tee -a "$LOG"
    exit 0
  fi
  n=$(score_n)
  echo "$(date +%Y-%m-%dT%H:%M:%S) attempt=$attempt scores=$n/360" | tee -a "$LOG"
  python3 -u scripts/gold_set/build_match_train_batch3.py >> "$LOG" 2>&1 &
  pid=$!
  last=$n
  stall=0
  while kill -0 "$pid" 2>/dev/null; do
    sleep 15
    cur=$(score_n)
    if [[ "$cur" -gt "$last" ]]; then
      last=$cur
      stall=0
      echo "$(date +%Y-%m-%dT%H:%M:%S) progress scores=$cur" | tee -a "$LOG"
    else
      stall=$((stall+15))
    fi
    if [[ "$stall" -ge 120 ]]; then
      echo "$(date +%Y-%m-%dT%H:%M:%S) STALL ${stall}s — kill $pid" | tee -a "$LOG"
      kill "$pid" 2>/dev/null || true
      sleep 2
      kill -9 "$pid" 2>/dev/null || true
      break
    fi
  done
  wait "$pid" 2>/dev/null || true
  if [[ -f data/processed/gold_sets/math_1_training_batch3/manifest.json ]]; then
    echo COMPLETE | tee -a "$LOG"
    exit 0
  fi
  sleep 1
done
echo FAILED_TOO_MANY_ATTEMPTS | tee -a "$LOG"
exit 1
