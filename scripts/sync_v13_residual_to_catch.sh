#!/usr/bin/env bash
# Sync v13 residual pack + v12 ckpt to Catch and start tmux train.
set -euo pipefail

HOST="${CATCH_SSH_TARGET:-catch@100.113.134.41}"
KEY="${CATCH_SSH_KEY:-$HOME/.ssh/id_ed25519_soccer_catch}"
SSH_E="ssh -i ${KEY} -o IdentitiesOnly=yes -o BatchMode=yes -o ConnectTimeout=20"
LOCAL_CV="/Volumes/LaCie/Projects/Soccer Coach CV"
REMOTE_CV="soccer_coach_cv"
PACK="/Volumes/LaCie/Projects/Soccer project data/ball_finetune_v13_residual"
TAR="/Volumes/LaCie/Projects/Soccer project data/ball_finetune_v13_residual.tar"

echo "=== 1) SSH Catch ==="
$SSH_E "$HOST" "echo connected; mkdir -p ~/${REMOTE_CV}/{datasets,models/v12_hard_snaps/post_train,configs,scripts,reports} ~/soccer_exchange/to_catch"

echo "=== 2) Tar pack ==="
export COPYFILE_DISABLE=1
if [[ ! -f "$TAR" ]]; then
  (cd "$PACK" && tar --disable-copyfile --no-xattrs -cf "$TAR" train valid manifest.json 2>/dev/null) || \
    (cd "$PACK" && tar -cf "$TAR" train valid manifest.json)
fi
rsync -avz --progress -e "$SSH_E" "$TAR" "$HOST:soccer_exchange/to_catch/ball_finetune_v13_residual.tar"

echo "=== 3) Unpack + configs on Catch ==="
$SSH_E "$HOST" bash -s <<REMOTE
set -euo pipefail
cd ~/${REMOTE_CV}
git pull --ff-only || true
source ~/.venvs/soccer-rfdetr312/bin/activate
rm -rf datasets/ball_finetune_v13_residual
mkdir -p datasets/ball_finetune_v13_residual
tar --no-same-owner -xf ~/soccer_exchange/to_catch/ball_finetune_v13_residual.tar -C datasets/ball_finetune_v13_residual
REMOTE

rsync -avz -e "$SSH_E" \
  "$LOCAL_CV/configs/finetune_v13_residual_catch.yaml" \
  "$LOCAL_CV/scripts/overnight_v13_residual.sh" \
  "$LOCAL_CV/scripts/train_ball.py" \
  "$HOST:${REMOTE_CV}/tmp_up_v13/" 2>/dev/null || \
rsync -avz -e "$SSH_E" \
  "$LOCAL_CV/configs/finetune_v13_residual_catch.yaml" \
  "$LOCAL_CV/scripts/overnight_v13_residual.sh" \
  "$LOCAL_CV/scripts/train_ball.py" \
  "$HOST:soccer_exchange/to_catch/"

$SSH_E "$HOST" bash -s <<'REMOTE'
set -euo pipefail
cd ~/soccer_coach_cv
mkdir -p configs scripts tmp_up_v13
cp -f ~/soccer_exchange/to_catch/finetune_v13_residual_catch.yaml configs/ 2>/dev/null || true
cp -f ~/soccer_exchange/to_catch/overnight_v13_residual.sh scripts/ 2>/dev/null || true
cp -f ~/soccer_exchange/to_catch/train_ball.py scripts/ 2>/dev/null || true
chmod +x scripts/overnight_v13_residual.sh
test -f models/v12_hard_snaps/post_train/checkpoint.pth || \
  echo "WARN: missing v12 checkpoint on Catch — push from Mac"
tmux has-session -t v13res 2>/dev/null && tmux kill-session -t v13res || true
tmux new-session -d -s v13res \
  "source ~/.venvs/soccer-rfdetr312/bin/activate; cd ~/soccer_coach_cv; \
   ROOT=~/soccer_coach_cv CFG=configs/finetune_v13_residual_catch.yaml \
   bash scripts/overnight_v13_residual.sh; bash"
echo STARTED_tmux_v13res
REMOTE

echo "Monitor: ssh -i $KEY $HOST 'tmux capture-pane -t v13res -p | tail -20'"
