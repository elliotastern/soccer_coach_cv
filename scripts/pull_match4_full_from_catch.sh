#!/usr/bin/env bash
# Mac: pull match_4_full batch output from Catch (events + frame_data per cam).
set -euo pipefail
KEY="${CATCH_SSH_KEY:-${HOME}/.ssh/id_ed25519_soccer_catch}"
HOST="${CATCH_SSH_TARGET:-catch@100.113.134.41}"
DEST="${1:-/Volumes/LaCie/Projects/Soccer Coach CV/data/output/match_4_full}"
REMOTE="soccer_coach_cv/data/output/match_4_full/"
SSH_E="ssh -i ${KEY} -o IdentitiesOnly=yes -o BatchMode=yes"
mkdir -p "${DEST}"
echo "Pulling ${HOST}:${REMOTE} → ${DEST}"
rsync -avz --progress -e "${SSH_E}" \
  "${HOST}:${REMOTE}" "${DEST}/"
echo "Done → ${DEST}"
