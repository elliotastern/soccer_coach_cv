#!/usr/bin/env bash
# Catch: after mosaic5 render finishes, run stride-4 eval @49s coach clip.
set -euo pipefail
KEY="${CATCH_SSH_KEY:-${HOME}/.ssh/id_ed25519_soccer_catch}"
HOST="${CATCH_SSH_TARGET:-catch@100.113.134.41}"
SSH="ssh -i ${KEY} -o IdentitiesOnly=yes -o BatchMode=yes ${HOST}"

${SSH} bash -s <<'REMOTE'
set -euo pipefail
if tmux has-session -t eval49_vid 2>/dev/null; then
  echo "eval49_vid already running"
  exit 0
fi
tmux new-session -d -s eval49_wait \
  'while tmux has-session -t mosaic5 2>/dev/null; do sleep 30; done; \
   tmux new-session -d -s eval49_vid \
   "source ~/.venvs/soccer-rfdetr312/bin/activate && cd ~/soccer_coach_cv && \
    export PYTHONPATH=. PYTHONUNBUFFERED=1 && \
    python3 scripts/gold_set/render_phase1_check_mosaic.py \
      --start 2940 --match-sec 15 --stride 4 --out-fps 4 \
      --out-dir reports/eval_match3/improve_eng_loop/real_fuse_eval_49s \
      --out-file coach_mosaic_eval_49s_stride4.mp4 \
      2>&1 | tee reports/eval_match3/improve_eng_loop/eval49_mosaic.log; \
    cp -f reports/eval_match3/improve_eng_loop/real_fuse_eval_49s/coach_mosaic_eval_49s_stride4.mp4 \
      ~/soccer_exchange/from_catch/coach_mosaic_eval_49s_stride4.mp4; \
    echo EVAL49_RENDER_DONE"; \
   tmux kill-session -t eval49_wait 2>/dev/null || true'
echo "Queued eval49_vid after mosaic5"
REMOTE
