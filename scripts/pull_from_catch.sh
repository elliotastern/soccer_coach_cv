#!/usr/bin/env bash
# Mac: pull ~/soccer_exchange/from_catch on Catch → local Downloads.
set -euo pipefail

KEY="${CATCH_SSH_KEY:-${HOME}/.ssh/id_ed25519_soccer_catch}"
HOST="${CATCH_SSH_TARGET:-catch-soccer}"
# Fallback if ~/.ssh/config Host catch-soccer not set (Tailscale IP from docs)
if ! ssh -i "${KEY}" -o IdentitiesOnly=yes -o BatchMode=yes -o ConnectTimeout=5 \
    "${HOST}" true 2>/dev/null; then
  HOST="${CATCH_SSH_FALLBACK:-catch@100.113.134.41}"
  echo "Using fallback host: ${HOST}"
fi
DEST="${1:-${HOME}/Downloads/soccer_catch_sync}"
REMOTE_DIR="soccer_exchange/from_catch/"
SSH_E="ssh -i ${KEY} -o IdentitiesOnly=yes -o BatchMode=yes"

mkdir -p "${DEST}"
echo "Pulling ${HOST}:${REMOTE_DIR} → ${DEST}"
rsync -avz --progress -e "${SSH_E}" "${HOST}:${REMOTE_DIR}" "${DEST}/"
echo "Done → ${DEST}"
