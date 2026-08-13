#!/usr/bin/env bash
# Sync v10 mix to RunPod and start train from v9 ckpt.
set -euo pipefail

HOST="${HOST:-root@69.30.85.39}"
PORT="${PORT:-22199}"
KEY="${KEY:-$HOME/.ssh/id_ed25519_runpod}"
LOCAL_CV="/Volumes/LaCie/Projects/Soccer Coach CV"
REMOTE="/workspace/soccer_cv_ball"
SSH=(ssh -T -o BatchMode=yes -o StrictHostKeyChecking=accept-new -i "$KEY" -p "$PORT" "$HOST")
SCP=(scp -o BatchMode=yes -o StrictHostKeyChecking=accept-new -i "$KEY" -P "$PORT")

echo "=== 1) SSH ==="
"${SSH[@]}" "echo connected; mkdir -p $REMOTE/{datasets,models/v9_snaps/post_train,configs,scripts,reports,tmp_up}"

echo "=== 2) Upload v10 dataset ==="
export COPYFILE_DISABLE=1
cd "/Volumes/LaCie/Projects/Soccer project data/ball_finetune_match2_v10"
tar cf - train valid test manifest.json | "${SSH[@]}" \
  "mkdir -p $REMOTE/datasets/ball_finetune_match2_v10 && tar --no-same-owner -xf - -C $REMOTE/datasets/ball_finetune_match2_v10"

echo "=== 3) Upload config + scripts ==="
"${SCP[@]}" \
  "$LOCAL_CV/configs/finetune_match2_v10.yaml" \
  "$LOCAL_CV/scripts/overnight_v10_match2.sh" \
  "$LOCAL_CV/scripts/train_ball.py" \
  "$LOCAL_CV/scripts/gold_set/eval_poc_ball_metrics.py" \
  "$HOST:$REMOTE/tmp_up/"
"${SSH[@]}" "cp $REMOTE/tmp_up/finetune_match2_v10.yaml $REMOTE/configs/; cp $REMOTE/tmp_up/overnight_v10_match2.sh $REMOTE/scripts/; cp $REMOTE/tmp_up/train_ball.py $REMOTE/scripts/; mkdir -p $REMOTE/scripts/gold_set; cp $REMOTE/tmp_up/eval_poc_ball_metrics.py $REMOTE/scripts/gold_set/; chmod +x $REMOTE/scripts/overnight_v10_match2.sh"

echo "=== 4) Start train ==="
"${SSH[@]}" "bash -s" <<'REMOTE_SCRIPT'
set -euo pipefail
cd /workspace/soccer_cv_ball
source venv/bin/activate
test -f models/v9_snaps/post_train/checkpoint.pth
mkdir -p models/dataset models/v10_snaps reports
ln -sfn /workspace/soccer_cv_ball/datasets/ball_finetune_match2_v10/train models/dataset/train
ln -sfn /workspace/soccer_cv_ball/datasets/ball_finetune_match2_v10/valid models/dataset/valid
ln -sfn /workspace/soccer_cv_ball/datasets/ball_finetune_match2_v10/test models/dataset/test
pkill -f train_ball.py || true
export PYTHONUNBUFFERED=1
nohup bash scripts/overnight_v10_match2.sh > reports/overnight_v10_match2.nohup.log 2>&1 &
echo STARTED_PID:$!
sleep 8
pgrep -af train_ball || echo "WARN: train_ball not in process list yet"
REMOTE_SCRIPT

echo "Monitor: ssh -i $KEY -p $PORT $HOST 'tail -f /workspace/soccer_cv_ball/reports/training_finetune_match2_v10.log'"
