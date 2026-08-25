#!/usr/bin/env bash
# Mac: pull a fuse eval clip folder from Catch (timeline + emits after GPU build).
set -euo pipefail
CLIP_ID="${1:-real_fuse_eval_49s}"
KEY="${CATCH_SSH_KEY:-${HOME}/.ssh/id_ed25519_soccer_catch}"
HOST="${CATCH_SSH_TARGET:-catch@100.113.134.41}"
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
REMOTE="soccer_coach_cv/data/processed/gold_sets/match3_events_v2_dribble/clips/${CLIP_ID}/"
DEST="${ROOT}/data/processed/gold_sets/match3_events_v2_dribble/clips/${CLIP_ID}/"
SSH_E="ssh -i ${KEY} -o IdentitiesOnly=yes -o BatchMode=yes"
mkdir -p "${DEST}"
echo "Pulling ${HOST}:${REMOTE} → ${DEST}"
rsync -avz --progress -e "${SSH_E}" \
  --exclude 'labels.json' \
  "${HOST}:${REMOTE}" "${DEST}/"
MANIFEST_REMOTE="soccer_coach_cv/data/processed/gold_sets/match3_events_v2_dribble/manifest.json"
if rsync -avz -e "${SSH_E}" "${HOST}:${MANIFEST_REMOTE}" \
  "${ROOT}/data/processed/gold_sets/match3_events_v2_dribble/manifest.json" 2>/dev/null; then
  echo "Updated manifest.json"
else
  echo "SKIP manifest (not on Catch — update Mac manifest locally)"
fi
echo "Done → ${DEST}"
