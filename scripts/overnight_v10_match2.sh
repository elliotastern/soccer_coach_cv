#!/usr/bin/env bash
# Resume v9 on Match2 v10 mix (RunPod paths). Short run to limit overfit.
set -euo pipefail
ROOT="${ROOT:-/workspace/soccer_cv_ball}"
cd "$ROOT"
export PYTHONUNBUFFERED=1

CFG="${CFG:-configs/finetune_match2_v10.yaml}"
OUT="${OUT:-models/v10_snaps}"
mkdir -p "$OUT/post_train" reports/eval_match2_v10

echo "=== v10 Match2 train ==="
echo "cfg=$CFG"
python3 scripts/train_ball.py --config "$CFG" 2>&1 | tee "reports/training_finetune_match2_v10.log"

if [[ -f models/checkpoint.pth ]]; then
  cp -f models/checkpoint.pth "$OUT/post_train/checkpoint.pth"
fi
CKPT="$OUT/post_train/checkpoint.pth"
if [[ ! -f "$CKPT" ]]; then
  CKPT="models/checkpoint.pth"
fi

if [[ -f scripts/gold_set/eval_poc_ball_metrics.py ]]; then
  python3 scripts/gold_set/eval_poc_ball_metrics.py \
    --ball-checkpoint "$CKPT" \
    --gold-dir data/processed/gold_sets/match2_gold_frames \
    --strip-max 49 --require-ball-gt \
    --out reports/eval_match2_v10/poc_match2_gold.json || true
  if [[ -d data/processed/gold_sets/match1_1_100/gold ]]; then
    python3 scripts/gold_set/eval_poc_ball_metrics.py \
      --ball-checkpoint "$CKPT" \
      --gold-dir data/processed/gold_sets/match1_1_100 \
      --strip-max 49 --require-ball-gt \
      --out reports/eval_match2_v10/poc_gold100.json || true
  fi
fi
echo "=== v10 done ==="
