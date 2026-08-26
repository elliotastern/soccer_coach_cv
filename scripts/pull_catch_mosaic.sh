#!/usr/bin/env bash
# Mac: pull 5-min mosaic render from Catch (live path, not exchange).
set -euo pipefail
KEY="${CATCH_SSH_KEY:-${HOME}/.ssh/id_ed25519_soccer_catch}"
HOST="${CATCH_SSH_TARGET:-catch@100.113.134.41}"
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
REMOTE="soccer_coach_cv/reports/eval_match3/improve_eng_loop/match4_5min/"
DEST="${ROOT}/reports/eval_match3/improve_eng_loop/match4_5min/"
SSH_E="ssh -i ${KEY} -o IdentitiesOnly=yes -o BatchMode=yes"
mkdir -p "${DEST}"
echo "Pulling ${HOST}:${REMOTE} → ${DEST}"
rsync -avz --progress -e "${SSH_E}" \
  "${HOST}:${REMOTE}" "${DEST}/"
echo "Done → ${DEST}"
