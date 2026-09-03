#!/usr/bin/env bash
# Resume v12_hard on residual FN pack (epochs 255→315). Paths via CFG.
set -euo pipefail
ROOT="${ROOT:-/workspace/soccer_cv_ball}"
cd "$ROOT"
export PYTHONUNBUFFERED=1

CFG="${CFG:-configs/finetune_v13_residual.yaml}"
OUT="${OUT:-models/v13_residual_snaps}"
mkdir -p "$OUT/post_train" reports models/dataset

TRAIN_DIR=$(python3 -c "import yaml; from pathlib import Path; c=yaml.safe_load(Path('$CFG').read_text()); print(c['dataset']['coco_train_path'])")
VALID_DIR=$(python3 -c "import yaml; from pathlib import Path; c=yaml.safe_load(Path('$CFG').read_text()); print(c['dataset']['coco_val_path'])")
ln -sfn "$TRAIN_DIR" models/dataset/train
ln -sfn "$VALID_DIR" models/dataset/valid

echo "=== v13 residual specialty train ==="
echo "cfg=$CFG root=$ROOT train=$TRAIN_DIR"
python3 scripts/train_ball.py --config "$CFG" 2>&1 | tee "reports/training_finetune_v13_residual.log"

if [[ -f models/checkpoint.pth ]]; then
  cp -f models/checkpoint.pth "$OUT/post_train/checkpoint.pth"
elif [[ -f models/checkpoint_best_total.pth ]]; then
  cp -f models/checkpoint_best_total.pth "$OUT/post_train/checkpoint.pth"
fi
echo "=== v13 residual done ==="
