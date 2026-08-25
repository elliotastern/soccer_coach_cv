#!/usr/bin/env bash
# Mac: pull ~/soccer_exchange/from_catch on Catch → local Downloads.
set -euo pipefail

HOST="${CATCH_SSH_TARGET:-catch-soccer}"
DEST="${1:-${HOME}/Downloads/soccer_catch_sync}"
REMOTE_DIR="soccer_exchange/from_catch/"
KEY="${CATCH_SSH_KEY:-${HOME}/.ssh/id_ed25519_soccer_catch}"
SSH_E="ssh -i ${KEY} -o IdentitiesOnly=yes -o BatchMode=yes"

mkdir -p "${DEST}"
echo "Pulling ${HOST}:${REMOTE_DIR} → ${DEST}"
rsync -avz --progress -e "${SSH_E}" "${HOST}:${REMOTE_DIR}" "${DEST}/"
echo "Done → ${DEST}"
