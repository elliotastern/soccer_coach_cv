#!/usr/bin/env bash
# Resume v8 on Match2 v9 mix (RunPod paths).
set -euo pipefail
ROOT="${ROOT:-/workspace/soccer_cv_ball}"
cd "$ROOT"
export PYTHONUNBUFFERED=1

CFG="${CFG:-configs/finetune_match2_v9.yaml}"
RESUME="${RESUME:-models/v8_snaps/post_train/checkpoint.pth}"
OUT="${OUT:-models/v9_snaps}"
mkdir -p "$OUT" reports/eval_match2_v9

echo "=== v9 Match2 train ==="
echo "cfg=$CFG resume=$RESUME"
python3 scripts/train_ball.py --config "$CFG" 2>&1 | tee "reports/training_finetune_match2_v9.log"

# train_ball writes models/checkpoint.pth; keep a v9-named copy for eval.
mkdir -p "$OUT/post_train"
if [[ -f models/checkpoint.pth ]]; then
  cp -f models/checkpoint.pth "$OUT/post_train/checkpoint.pth"
fi
CKPT="${CKPT:-$OUT/post_train/checkpoint.pth}"
if [[ ! -f "$CKPT" ]]; then
  CKPT="models/checkpoint.pth"
fi

if [[ -f scripts/gold_set/eval_poc_ball_metrics.py ]]; then
  python3 scripts/gold_set/eval_poc_ball_metrics.py \
    --ball-checkpoint "$CKPT" \
    --gold-dir data/processed/gold_sets/match2_gold_frames \
    --strip-max 49 \
    --require-ball-gt \
    --out reports/eval_match2_v9/poc_match2_gold.json || true
fi
echo "=== v9 done ==="
