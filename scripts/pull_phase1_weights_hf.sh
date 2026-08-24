#!/usr/bin/env bash
# Download Phase 1 weights from private HF repo into models/.
# Uses HF_TOKEN if set, else saved token from hf auth login.
#
#   export HF_REPO=youruser/soccer-coach-phase1-weights
#   bash scripts/pull_phase1_weights_hf.sh
#
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
REPO="${HF_REPO:-eeeeeeeeeeeeee3/soccer-coach-phase1-weights}"

fail() { echo "ERROR: $*" >&2; exit 1; }

[[ -n "$REPO" ]] || fail "set HF_REPO=username/soccer-coach-phase1-weights"
command -v hf >/dev/null || pip install -q 'huggingface_hub>=0.30,<1'

mkdir -p models/v12_hard_snaps/post_train
echo "Downloading $REPO → models/"
if [[ -n "${HF_TOKEN:-}" ]]; then
  hf download "$REPO" --local-dir models --token "$HF_TOKEN"
else
  hf download "$REPO" --local-dir models
fi

test -f models/people_after_100_epochs.pth || fail "player .pth missing after download"
test -f models/v12_hard_snaps/post_train/checkpoint.pth || fail "ball .pth missing after download"
ls -lh models/people_after_100_epochs.pth models/v12_hard_snaps/post_train/checkpoint.pth
echo "OK — next: bash scripts/setup_catch_phase1_continue.sh"
