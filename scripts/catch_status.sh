#!/usr/bin/env bash
# One-line Catch job status (tmux + logs + exchange).
set -euo pipefail
KEY="${CATCH_SSH_KEY:-${HOME}/.ssh/id_ed25519_soccer_catch}"
HOST="${CATCH_SSH_TARGET:-catch@100.113.134.41}"
SSH="ssh -i ${KEY} -o IdentitiesOnly=yes -o BatchMode=yes ${HOST}"
LOG_ROOT="soccer_coach_cv/reports/eval_match3/improve_eng_loop"

${SSH} bash -s <<'REMOTE'
set -euo pipefail
echo "=== tmux ==="
tmux list-sessions 2>/dev/null || echo "(no tmux)"
for s in mosaic5 fuse_eval_queue match4_full mosaic_watch; do
  if tmux has-session -t "$s" 2>/dev/null; then
    echo "--- $s (last lines) ---"
    tmux capture-pane -t "$s" -p 2>/dev/null | tail -3
  fi
done
echo "=== logs ==="
for f in batch_match4_full.log match4_5min_rerender.log; do
  p="$HOME/soccer_coach_cv/reports/eval_match3/improve_eng_loop/$f"
  if [[ -f "$p" ]]; then tail -2 "$p"; else echo "missing $f"; fi
done
echo "=== match_4_full checkpoints ==="
ls -lt "$HOME/soccer_coach_cv/data/output/match_4_full/P10-match4/checkpoints/" 2>/dev/null | head -4 || true
echo "=== exchange ==="
ls -lh "$HOME/soccer_exchange/from_catch/" 2>/dev/null || true
REMOTE
