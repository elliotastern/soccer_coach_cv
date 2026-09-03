#!/usr/bin/env bash
# Poll Catch until SSH works, then start Match-4 fullcam kit 5-min batch in tmux.
# See docs/product/CATCH_MATCH4_KIT_CONSENSUS_9.md
#
# Launch (Mac): nohup env POLL_SEC=60 bash scripts/wait_catch_and_run_match4_fullcam_kit.sh >>/tmp/wait_catch_m4kit.log 2>&1 &
set -eu
CATCH="${CATCH_SSH_TARGET:-catch@100.113.134.41}"
KEY="${CATCH_SSH_KEY:-$HOME/.ssh/id_ed25519_soccer_catch}"
INTERVAL="${POLL_SEC:-60}"
MAX_TRIES="${MAX_TRIES:-0}"  # 0 = forever

echo "[wait] $(date -u +%Y-%m-%dT%H:%M:%SZ) start polling Catch=$CATCH interval=${INTERVAL}s"

n=0
while true; do
  n=$((n + 1))
  if ssh -o ConnectTimeout=8 -o BatchMode=yes -i "$KEY" "$CATCH" 'echo catch_ok' 2>/dev/null | grep -q catch_ok; then
    echo "[wait] Catch reachable after try $n"
    break
  fi
  echo "[wait] try $n: Catch not up yet ($(date -u +%H:%M:%SZ)); sleep ${INTERVAL}s"
  if [ "$MAX_TRIES" -gt 0 ] && [ "$n" -ge "$MAX_TRIES" ]; then
    echo "[wait] gave up after $MAX_TRIES tries" >&2
    exit 2
  fi
  sleep "$INTERVAL"
done

ssh -i "$KEY" "$CATCH" bash -s <<'REMOTE'
set -eu
cd ~/soccer_coach_cv
git pull --ff-only || git pull
# shellcheck disable=SC1090
source ~/.venvs/soccer-rfdetr312/bin/activate
if [ -f data/output/match_4_5min/team_centroids.json ]; then
  echo "[catch] kit-ref present"
else
  echo "[catch] WARNING: no team_centroids.json — label on 8503 before trusting kit>=9" >&2
fi
tmux has-session -t m4kit 2>/dev/null && tmux kill-session -t m4kit || true
tmux new-session -d -s m4kit "cd ~/soccer_coach_cv && source ~/.venvs/soccer-rfdetr312/bin/activate && CAMS='P10-match4 P9-match4 P7-match4 P8-match4 P1-match4 P6-match4' bash scripts/run_batch_match4_5min.sh 2>&1 | tee /tmp/m4_fullcam_kit.log"
echo "[catch] started tmux session m4kit"
tmux capture-pane -t m4kit -p | tail -5 || true
REMOTE

echo "[wait] Batch launched in Catch tmux m4kit."
echo "  ssh -i $KEY $CATCH 'tmux capture-pane -t m4kit -p | tail -20'"
