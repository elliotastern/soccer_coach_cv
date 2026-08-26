#!/usr/bin/env bash
# Mac: wait for Catch mosaic RENDER_DONE + eval clips, then pull exchange.
set -euo pipefail
KEY="${CATCH_SSH_KEY:-${HOME}/.ssh/id_ed25519_soccer_catch}"
HOST="${CATCH_SSH_TARGET:-catch@100.113.134.41}"
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
SSH="ssh -i ${KEY} -o IdentitiesOnly=yes -o BatchMode=yes ${HOST}"
INTERVAL="${POLL_SEC:-60}"
MAX_ROUNDS="${MAX_ROUNDS:-90}"

for ((i=1; i<=MAX_ROUNDS; i++)); do
  status="$(${SSH} bash -s <<'REMOTE'
LOG=~/soccer_coach_cv/reports/eval_match3/improve_eng_loop/match4_5min_rerender.log
MOS=$(tmux has-session -t mosaic5 2>/dev/null && echo MOSAIC_UP || echo MOSAIC_DONE)
RD=$(grep -q RENDER_DONE "$LOG" 2>/dev/null && echo RENDER_DONE || echo NO)
EQ=$(tmux has-session -t fuse_eval_queue 2>/dev/null && echo QUEUE_UP || echo QUEUE_DONE)
EV=$(ls ~/soccer_exchange/from_catch/coach_mosaic_eval_49s_stride4.mp4 2>/dev/null && echo EVAL49 || echo NO49)
EV69=$(ls ~/soccer_exchange/from_catch/coach_mosaic_eval_69s_stride4.mp4 2>/dev/null && echo EVAL69 || echo NO69)
FR=$(tmux capture-pane -t mosaic5 -p 2>/dev/null | tail -1)
echo "$MOS $RD $EQ $EV $EV69 | $FR"
REMOTE
)"
  echo "$(date +%H:%M:%S) $status"
  if echo "$status" | grep -q "RENDER_DONE"; then
    bash "${ROOT}/scripts/pull_from_catch.sh"
    bash "${ROOT}/scripts/pull_catch_mosaic.sh"
  fi
  if echo "$status" | grep -q "EVAL49"; then
    bash "${ROOT}/scripts/pull_from_catch.sh"
    mkdir -p "${ROOT}/reports/eval_match3/improve_eng_loop/real_fuse_eval_49s"
    cp -f "${HOME}/Downloads/soccer_catch_sync/"*eval_49s* \
      "${ROOT}/reports/eval_match3/improve_eng_loop/real_fuse_eval_49s/" 2>/dev/null || true
  fi
  if echo "$status" | grep -qE "MOSAIC_DONE.*QUEUE_DONE.*EVAL49"; then
    echo "All staged — final pull"
    bash "${ROOT}/scripts/pull_from_catch.sh"
    bash "${ROOT}/scripts/pull_match4_full_from_catch.sh"
    python3 "${ROOT}/scripts/gold_set/audit_batch_events_pack.py" | tail -5
    break
  fi
  sleep "${INTERVAL}"
done
