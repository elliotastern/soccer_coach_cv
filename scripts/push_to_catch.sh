#!/usr/bin/env bash
# Mac: push file(s) to Catch ~/soccer_exchange/to_catch/
set -euo pipefail

HOST="${CATCH_SSH_TARGET:-catch-soccer}"
REMOTE_DIR="soccer_exchange/to_catch"
KEY="${CATCH_SSH_KEY:-${HOME}/.ssh/id_ed25519_soccer_catch}"
SSH_E="ssh -i ${KEY} -o IdentitiesOnly=yes -o BatchMode=yes"

if [[ $# -lt 1 ]]; then
  echo "Usage: bash scripts/push_to_catch.sh <file-or-dir> [more…]"
  echo "Set CATCH_SSH_TARGET=catch@100.x.x.x if ~/.ssh/config missing catch-soccer"
  exit 1
fi

for src in "$@"; do
  [[ -e "$src" ]] || { echo "missing: $src"; exit 1; }
  echo "Pushing $src → ${HOST}:${REMOTE_DIR}/"
  rsync -avz --progress -e "${SSH_E}" "$src" "${HOST}:${REMOTE_DIR}/"
done
echo "On Catch: ls ~/soccer_exchange/to_catch/"
