#!/usr/bin/env bash
# Poll Catch until online, then run Match 4 team ID grid + stage results.
set -euo pipefail
HOST="${CATCH_SSH_TARGET:-catch@100.113.134.41}"
KEY="${CATCH_SSH_KEY:-${HOME}/.ssh/id_ed25519_soccer_catch}"
SSH_E="ssh -i ${KEY} -o IdentitiesOnly=yes -o BatchMode=yes -o ConnectTimeout=10"
REPO="${CATCH_REPO:-~/soccer_coach_cv}"
MAX_WAIT_MIN="${MAX_WAIT_MIN:-180}"
INTERVAL_SEC="${INTERVAL_SEC:-60}"

echo "Waiting for Catch (${HOST}) up to ${MAX_WAIT_MIN} min…"
deadline=$((SECONDS + MAX_WAIT_MIN * 60))
until ${SSH_E} "${HOST}" "echo ok" 2>/dev/null; do
  if (( SECONDS >= deadline )); then
    echo "Catch still offline after ${MAX_WAIT_MIN} min."
    exit 1
  fi
  sleep "${INTERVAL_SEC}"
done

echo "Catch online — syncing grid scripts…"
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
rsync -avz -e "${SSH_E}" \
  "${ROOT}/scripts/eval_team_id_strategy_grid.py" \
  "${ROOT}/scripts/apply_team_id_grid_winner.py" \
  "${ROOT}/scripts/run_team_id_grid_on_catch.sh" \
  "${ROOT}/src/perception/team_strategy.py" \
  "${ROOT}/src/perception/team_core.py" \
  "${ROOT}/src/review/team_live.py" \
  "${ROOT}/src/review/multicam_fuse.py" \
  "${HOST}:${REPO}/"

${SSH_E} "${HOST}" "bash ${REPO}/scripts/run_team_id_grid_on_catch.sh ${REPO}"
echo "Grid started in tmux on Catch. Pull when done:"
echo "  rsync -avz -e \"${SSH_E}\" ${HOST}:${REPO}/reports/eval_match3/team_id_strategy_grid/ reports/eval_match3/team_id_strategy_grid_m4/"
