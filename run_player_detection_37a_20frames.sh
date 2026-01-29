#!/usr/bin/env bash
# Run player (person) detection on 37a video for 20 frames.
# Model: models/checkpoint_best_total_after_100_epochs.pth
set -e
cd "$(dirname "$0")"
python scripts/process_video_pipeline.py \
  "data/raw/37CAE053-841F-4851-956E-CBF17A51C506.mp4" \
  --model models/checkpoint_best_total_after_100_epochs.pth \
  --output data/output/37a_20frames \
  --max-frames 20
