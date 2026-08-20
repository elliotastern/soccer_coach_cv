#!/usr/bin/env bash
# Upload ball_tracknet_seq_v1 + TrackNet trainer to RunPod and start train.
set -euo pipefail

HOST="${HOST:-runpod-tcp}"
KEY="${KEY:-$HOME/.ssh/id_ed25519_runpod}"
LOCAL_CV="/Volumes/LaCie/Projects/Soccer Coach CV"
PACK_SRC="/Volumes/LaCie/Projects/Soccer project data/ball_tracknet_seq_v1"
TAR="/Volumes/LaCie/Projects/Soccer project data/ball_tracknet_seq_v1.tar"
REMOTE="/workspace/soccer_cv_ball"
SSH=(ssh -T -o BatchMode=yes -o StrictHostKeyChecking=accept-new -i "$KEY" "$HOST")
SCP=(scp -o BatchMode=yes -o StrictHostKeyChecking=accept-new -o Compression=no -i "$KEY")

echo "=== 1) SSH ==="
"${SSH[@]}" "echo connected; mkdir -p $REMOTE/{datasets,models,scripts/tracknet,reports,tmp_up}"

echo "=== 2) Build tar (follow symlinks; skip AppleDouble) ==="
export COPYFILE_DISABLE=1
if [[ ! -f "$TAR" ]] || [[ "$PACK_SRC/manifest.json" -nt "$TAR" ]]; then
  echo "Building $TAR ..."
  (
    cd "$PACK_SRC"
    tar --disable-copyfile --no-xattrs -h -cf "$TAR" \
      --exclude='._*' \
      clips splits heatmaps manifest.json
  )
fi
ls -lh "$TAR"

echo "=== 3) Upload pack tar ==="
"${SCP[@]}" "$TAR" "$HOST:$REMOTE/tmp_up/ball_tracknet_seq_v1.tar"
"${SSH[@]}" "rm -rf $REMOTE/datasets/ball_tracknet_seq_v1 && mkdir -p $REMOTE/datasets/ball_tracknet_seq_v1 && tar --no-same-owner -xf $REMOTE/tmp_up/ball_tracknet_seq_v1.tar -C $REMOTE/datasets/ball_tracknet_seq_v1 && rm -f $REMOTE/tmp_up/ball_tracknet_seq_v1.tar && python3 - <<'PY'
from pathlib import Path
p=Path('/workspace/soccer_cv_ball/datasets/ball_tracknet_seq_v1')
print('clips', list((p/'clips').iterdir()))
print('train lines', sum(1 for _ in open(p/'splits'/'train_triplets.jsonl')))
print('frame0', (p/'clips'/'match2_4quad_top_left'/'frames'/'0000.jpg').is_file())
PY"

echo "=== 4) Upload trainer ==="
"${SCP[@]}" \
  "$LOCAL_CV/scripts/tracknet/model.py" \
  "$LOCAL_CV/scripts/tracknet/dataset.py" \
  "$LOCAL_CV/scripts/tracknet/train.py" \
  "$LOCAL_CV/scripts/tracknet/eval_pack.py" \
  "$LOCAL_CV/scripts/overnight_tracknet_seq.sh" \
  "$HOST:$REMOTE/tmp_up/"
"${SSH[@]}" "cp $REMOTE/tmp_up/{model,dataset,train,eval_pack}.py $REMOTE/scripts/tracknet/; cp $REMOTE/tmp_up/overnight_tracknet_seq.sh $REMOTE/scripts/; chmod +x $REMOTE/scripts/overnight_tracknet_seq.sh; touch $REMOTE/scripts/tracknet/__init__.py"

echo "=== 5) Smoke + start train ==="
"${SSH[@]}" "bash -s" <<'REMOTE_SCRIPT'
set -euo pipefail
cd /workspace/soccer_cv_ball
python3 - <<'PY'
from pathlib import Path
import sys
sys.path.insert(0, 'scripts/tracknet')
from dataset import SeqTripletDataset
from model import TrackNetV2
import torch
ds = SeqTripletDataset(Path('datasets/ball_tracknet_seq_v1'), 'train')
b = ds[0]
print('smoke x', tuple(b['x'].shape), 'y', tuple(b['y'].shape), 'vis', int(b['visible']))
m = TrackNetV2()
with torch.no_grad():
    y = m(b['x'].unsqueeze(0))
print('out', tuple(y.shape))
PY
pkill -f 'scripts/tracknet/train.py' || true
nohup bash scripts/overnight_tracknet_seq.sh > reports/tracknet_seq_v1.nohup.log 2>&1 &
echo STARTED_PID:$!
sleep 12
pgrep -af 'tracknet/train.py' || echo 'WARN: train not listed yet'
tail -n 50 reports/tracknet_seq_v1.nohup.log || true
REMOTE_SCRIPT

echo "Monitor: ssh $HOST 'tail -f /workspace/soccer_cv_ball/reports/tracknet_seq_v1.log'"
