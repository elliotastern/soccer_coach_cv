#!/usr/bin/env bash
set -uo pipefail
cd /workspace/soccer_cv_ball
. venv/bin/activate
export NO_ALBUMENTATIONS_UPDATE=1 PYTHONUNBUFFERED=1
ST=overnight_v7_STATUS.txt
echo "$(date -u +%Y-%m-%dT%H:%M:%SZ) V7 START" | tee "$ST"

if [ -f datasets/ball_finetune_match_mix_v7.tar.gz ] && [ ! -d datasets/ball_finetune_match_mix_v7/train ]; then
  echo "$(date -u +%Y-%m-%dT%H:%M:%SZ) extract v7 pack" | tee -a "$ST"
  tar --no-same-owner -xzf datasets/ball_finetune_match_mix_v7.tar.gz -C datasets/
fi
if [ ! -d datasets/ball_finetune_match_mix_v7/train ]; then
  echo "MISSING v7 dataset" | tee -a "$ST"
  echo FAIL > overnight_v7_FAILED
  exit 1
fi

# Prefer Gold-best v6 epoch_110 regular weights
RESUME=/workspace/soccer_cv_ball/models/v6_snaps/epoch_110/checkpoint_best_regular.pth
if [ ! -f "$RESUME" ]; then
  RESUME=/workspace/soccer_cv_ball/models/v6_snaps/final/checkpoint_best_regular.pth
fi
if [ ! -f "$RESUME" ]; then
  RESUME=/workspace/soccer_cv_ball/models/checkpoint_best_regular.pth
fi
echo "$(date -u +%Y-%m-%dT%H:%M:%SZ) resume=$RESUME" | tee -a "$ST"

rm -rf models/dataset
mkdir -p models/dataset models/v7_snaps
ln -sfn /workspace/soccer_cv_ball/datasets/ball_finetune_match_mix_v7/train models/dataset/train
ln -sfn /workspace/soccer_cv_ball/datasets/ball_finetune_match_mix_v7/valid models/dataset/valid
ln -sfn /workspace/soccer_cv_ball/datasets/ball_finetune_match_mix_v7/test models/dataset/test

# Ensure test has images (RF-DETR sometimes wants test/)
if [ -d models/dataset/valid ] && [ -d models/dataset/test ]; then
  # if test empty of jpgs, copy valid
  ntest=$(ls models/dataset/test/*.jpg 2>/dev/null | wc -l || echo 0)
  if [ "${ntest:-0}" -lt 1 ]; then
    cp -a models/dataset/valid/. models/dataset/test/ || true
  fi
fi

python3 - <<PY
from pathlib import Path
import re
resume = Path("$RESUME")
p = Path("configs/finetune_match_mix_v7.yaml")
t = p.read_text()
t = re.sub(r"(?m)^  resume_from:.*$", f"  resume_from: {resume}", t)
t = re.sub(r"(?m)^  epochs:.*$", "  epochs: 160", t)
# point dataset paths are already v7 in config; also ensure train_ball uses models/dataset via prepare
p.write_text(t)
print("config resume", resume, "exists", resume.is_file())
print("train imgs", len(list(Path("models/dataset/train").glob("*.jpg"))))
print("valid imgs", len(list(Path("models/dataset/valid").glob("*.jpg"))))
print("test imgs", len(list(Path("models/dataset/test").glob("*.jpg"))))
PY

# Dense watcher for v7
if [ -f scripts/v6_dense_watcher.py ]; then
  sed 's/v6_snaps/v7_snaps/g; s/training_finetune_match_mix_v6/training_finetune_match_mix_v7/g; s/overnight_v6_STOP_WATCH/overnight_v7_STOP_WATCH/g; s/v6 dense/v7 dense/g' \
    scripts/v6_dense_watcher.py > scripts/v7_dense_watcher.py
  rm -f overnight_v7_STOP_WATCH
  python3 -u scripts/v7_dense_watcher.py > overnight_v7_watch.log 2>&1 &
  echo "watcher_pid=$!" | tee -a "$ST"
fi

# Optional gold sweep watcher if present
if [ -f scripts/gold_sweep_watcher.py ]; then
  # leave alone; v6 gold sweep may have finished
  :
fi

echo "$(date -u +%Y-%m-%dT%H:%M:%SZ) TRAIN v7 begin pack=v7 resume_gold_best" | tee -a "$ST"
set +e
python -u scripts/train_ball.py --config configs/finetune_match_mix_v7.yaml --output-dir models \
  > training_finetune_match_mix_v7.log 2>&1
RC=$?
set -e
echo "$(date -u +%Y-%m-%dT%H:%M:%SZ) TRAIN v7 rc=$RC" | tee -a "$ST"
mkdir -p models/v7_snaps/final
cp -f models/checkpoint.pth models/checkpoint_best_regular.pth models/checkpoint_best_ema.pth models/v7_snaps/final/ 2>/dev/null || true
rm -f models/checkpoint0*.pth 2>/dev/null || true
touch overnight_v7_STOP_WATCH 2>/dev/null || true
if [ "$RC" -eq 0 ]; then echo COMPLETE > overnight_v7_COMPLETE; else echo FAIL > overnight_v7_FAILED; fi
echo "$(date -u +%Y-%m-%dT%H:%M:%SZ) DONE — stop pod in dashboard" | tee -a "$ST"
