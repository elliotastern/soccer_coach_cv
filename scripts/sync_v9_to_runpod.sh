#!/usr/bin/env bash
# Sync v9 mix + v8 ckpt to RunPod and start overnight train.
# Direct TCP SSH (proxy ssh.runpod.io does not support scp).
#   PORT/IP change per pod — read from the pod: echo $RUNPOD_PUBLIC_IP $RUNPOD_TCP_PORT_22
set -euo pipefail

HOST="${HOST:-root@69.30.85.39}"
PORT="${PORT:-22199}"
KEY="${KEY:-$HOME/.ssh/id_ed25519_runpod}"
LOCAL_CV="/Volumes/LaCie/Projects/Soccer Coach CV"
REMOTE="/workspace/soccer_cv_ball"
CKPT_LOCAL="$LOCAL_CV/models/v8_snaps/post_train/checkpoint.pth"
SSH=(ssh -T -o BatchMode=yes -o StrictHostKeyChecking=accept-new -i "$KEY" -p "$PORT" "$HOST")
SCP=(scp -o BatchMode=yes -o StrictHostKeyChecking=accept-new -i "$KEY" -P "$PORT")

echo "=== 1) Check SSH ==="
"${SSH[@]}" "echo connected; mkdir -p $REMOTE/{datasets,models/v8_snaps/post_train,configs,scripts,reports}"

echo "=== 2) Upload v9 dataset (no _match_aug) ==="
export COPYFILE_DISABLE=1
cd "/Volumes/LaCie/Projects/Soccer project data/ball_finetune_match2_v9"
tar cf - train valid test manifest.json | "${SSH[@]}" \
  "mkdir -p $REMOTE/datasets/ball_finetune_match2_v9 && tar --no-same-owner -xf - -C $REMOTE/datasets/ball_finetune_match2_v9"

echo "=== 3) Upload v8 checkpoint ==="
"${SCP[@]}" "$CKPT_LOCAL" "$HOST:$REMOTE/models/v8_snaps/post_train/checkpoint.pth"

echo "=== 4) Upload config + scripts ==="
"${SCP[@]}" \
  "$LOCAL_CV/configs/finetune_match2_v9.yaml" \
  "$LOCAL_CV/scripts/overnight_v9_match2.sh" \
  "$LOCAL_CV/scripts/train_ball.py" \
  "$HOST:$REMOTE/tmp_up/"
"${SSH[@]}" "mkdir -p $REMOTE/tmp_up; cp $REMOTE/tmp_up/finetune_match2_v9.yaml $REMOTE/configs/; cp $REMOTE/tmp_up/overnight_v9_match2.sh $REMOTE/scripts/; cp $REMOTE/tmp_up/train_ball.py $REMOTE/scripts/; chmod +x $REMOTE/scripts/overnight_v9_match2.sh"

echo "=== 5) Start overnight train ==="
"${SSH[@]}" "bash -s" <<'REMOTE_SCRIPT'
set -euo pipefail
cd /workspace/soccer_cv_ball
source venv/bin/activate
mkdir -p models/dataset models/v9_snaps reports
ln -sfn /workspace/soccer_cv_ball/datasets/ball_finetune_match2_v9/train models/dataset/train
ln -sfn /workspace/soccer_cv_ball/datasets/ball_finetune_match2_v9/valid models/dataset/valid
ln -sfn /workspace/soccer_cv_ball/datasets/ball_finetune_match2_v9/test models/dataset/test
chmod +x scripts/overnight_v9_match2.sh
pkill -f train_ball.py || true
export PYTHONUNBUFFERED=1
nohup bash scripts/overnight_v9_match2.sh > reports/overnight_v9_match2.nohup.log 2>&1 &
echo STARTED_PID:$!
sleep 5
pgrep -af train_ball || echo "WARN: train_ball not in process list yet"
REMOTE_SCRIPT

echo "Monitor: ssh -i $KEY -p $PORT $HOST 'tail -f /workspace/soccer_cv_ball/reports/training_finetune_match2_v9.log'"
