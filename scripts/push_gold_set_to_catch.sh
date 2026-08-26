#!/usr/bin/env bash
# Mac: push gold_set manifest + clip labels to Catch (no large timelines).
set -euo pipefail
KEY="${CATCH_SSH_KEY:-${HOME}/.ssh/id_ed25519_soccer_catch}"
HOST="${CATCH_SSH_TARGET:-catch@100.113.134.41}"
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
SSH_E="ssh -i ${KEY} -o IdentitiesOnly=yes -o BatchMode=yes"
REMOTE_BASE="soccer_coach_cv/data/processed/gold_sets/match3_events_v2_dribble"
${SSH_E} "${HOST}" "mkdir -p ${REMOTE_BASE}/clips"
rsync -avz -e "${SSH_E}" \
  "${ROOT}/data/processed/gold_sets/match3_events_v2_dribble/manifest.json" \
  "${HOST}:${REMOTE_BASE}/manifest.json"
for clip in real_fuse_eval_49s real_fuse_eval_69s real_fuse_eval_20s real_fuse_holdout_pass; do
  labels="${ROOT}/data/processed/gold_sets/match3_events_v2_dribble/clips/${clip}/labels.json"
  if [[ -f "${labels}" ]]; then
    ${SSH_E} "${HOST}" "mkdir -p ${REMOTE_BASE}/clips/${clip}"
    rsync -avz -e "${SSH_E}" "${labels}" \
      "${HOST}:${REMOTE_BASE}/clips/${clip}/labels.json"
  fi
done
echo "Pushed manifest + labels → ${HOST}:${REMOTE_BASE}"
