#!/usr/bin/env bash
# Train TrackNetV2-style on ball_tracknet_seq_v1 (RunPod).
set -euo pipefail
ROOT="${ROOT:-/workspace/soccer_cv_ball}"
cd "$ROOT"
export PYTHONUNBUFFERED=1
PACK="$ROOT/datasets/ball_tracknet_seq_v1"
OUT="$ROOT/models/tracknet_seq_v1"
mkdir -p "$OUT" reports
echo "=== tracknet_seq_v1 start $(date -u +%Y-%m-%dT%H:%M:%SZ) ===" | tee -a reports/tracknet_seq_v1.log
python3 scripts/tracknet/train.py \
  --pack "$PACK" \
  --out "$OUT" \
  --epochs 60 \
  --batch 8 \
  --lr 0.001 \
  --workers 4 \
  --tol 4 \
  2>&1 | tee -a reports/tracknet_seq_v1.log
echo "=== tracknet_seq_v1 done $(date -u +%Y-%m-%dT%H:%M:%SZ) ===" | tee -a reports/tracknet_seq_v1.log
