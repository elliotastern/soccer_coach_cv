#!/usr/bin/env bash
# Upload Phase 1 RF-DETR weights to a private Hugging Face model repo.
# Run on DEVELOPER Mac only. Requires HF write token.
#
#   export HF_TOKEN=hf_...          # write token (never commit)
#   export HF_REPO=youruser/soccer-coach-phase1-weights
#   bash scripts/push_phase1_weights_hf.sh
#
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

PLAYER="$ROOT/models/people_after_100_epochs.pth"
BALL="$ROOT/models/v12_hard_snaps/post_train/checkpoint.pth"
REPO="${HF_REPO:-}"

fail() { echo "ERROR: $*" >&2; exit 1; }

[[ -n "${HF_TOKEN:-}" ]] || fail "set HF_TOKEN (write token) — do not paste into chat"
[[ -n "$REPO" ]] || fail "set HF_REPO=username/soccer-coach-phase1-weights"
[[ -f "$PLAYER" ]] || fail "missing $PLAYER"
[[ -f "$BALL" ]] || fail "missing $BALL"
command -v hf >/dev/null || fail "hf CLI missing — pip install 'huggingface_hub>=0.30,<1'"

echo "Uploading to private model repo: $REPO"
hf upload "$REPO" "$PLAYER" people_after_100_epochs.pth \
  --repo-type model --private --token "$HF_TOKEN" \
  --commit-message "Phase 1 player RF-DETR weights"

hf upload "$REPO" "$BALL" v12_hard_snaps/post_train/checkpoint.pth \
  --repo-type model --private --token "$HF_TOKEN" \
  --commit-message "Phase 1 ball v12_hard weights"

echo "OK — https://huggingface.co/$REPO"
echo "On Catch: export HF_TOKEN=... HF_REPO=$REPO && bash scripts/pull_phase1_weights_hf.sh"
