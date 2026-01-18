#!/bin/bash
# Resume training with class weighting and fixed label mapping
# Uses lower learning rate for fine-tuning

cd "$(dirname "$0")/.."

echo "Resuming training from checkpoint with fixes:"
echo "  - Fixed label mapping"
echo "  - Class weighting enabled (ball: 25x)"
echo "  - Per-class metrics tracking"
echo "  - Lower learning rate for fine-tuning"
echo ""

# Resume from latest checkpoint
python scripts/train_detr.py \
    --config configs/training.yaml \
    --train-dir datasets/train \
    --val-dir datasets/val \
    --output-dir models \
    --resume models/checkpoints/latest_checkpoint.pth
