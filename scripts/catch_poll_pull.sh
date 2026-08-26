#!/usr/bin/env bash
# Mac: poll Catch batch promote + pull match_4_full when frame count advances.
set -euo pipefail
KEY="${CATCH_SSH_KEY:-${HOME}/.ssh/id_ed25519_soccer_catch}"
HOST="${CATCH_SSH_TARGET:-catch@100.113.134.41}"
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
SSH="ssh -i ${KEY} -o IdentitiesOnly=yes -o BatchMode=yes ${HOST}"
LAST="${ROOT}/reports/events_testing/.match4_full_pull_frames"
INTERVAL="${POLL_SEC:-45}"
MAX_ROUNDS="${MAX_ROUNDS:-120}"

read_last() {
  [[ -f "$LAST" ]] && cat "$LAST" || echo 0
}

get_frames() {
  ${SSH} "python3 -c \"
import json,pandas as pd
from pathlib import Path
p=Path.home()/'soccer_coach_cv/data/output/match_4_full/P10-match4_cumulative/frame_data.csv'
if p.is_file():
  print(len(pd.read_csv(p)['frame_id'].unique()))
else:
  print(0)
\"" 2>/dev/null || echo 0
}

write_last() { echo "$1" > "$LAST"; }

echo "Polling Catch batch → pull when frames advance (every ${INTERVAL}s)"
cur="$(read_last)"
for ((i=1; i<=MAX_ROUNDS; i++)); do
  frames="$(get_frames)"
  mosaic="$( ${SSH} 'tmux capture-pane -t mosaic5 -p 2>/dev/null | tail -1' || true)"
  echo "$(date +%H:%M:%S) frames=$frames last=$cur mosaic: ${mosaic:-n/a}"
  if [[ "${frames}" -gt "${cur}" ]]; then
    bash "${ROOT}/scripts/pull_match4_full_from_catch.sh"
    python3 "${ROOT}/scripts/gold_set/audit_batch_events_pack.py" | tail -3
    write_last "${frames}"
    cur="${frames}"
  fi
  if ${SSH} 'grep -q RENDER_DONE ~/soccer_coach_cv/reports/eval_match3/improve_eng_loop/match4_5min_rerender.log 2>/dev/null'; then
    echo "mosaic RENDER_DONE — pull exchange"
    bash "${ROOT}/scripts/pull_from_catch.sh"
    bash "${ROOT}/scripts/pull_catch_mosaic.sh"
    break
  fi
  sleep "${INTERVAL}"
done
