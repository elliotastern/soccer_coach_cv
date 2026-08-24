#!/usr/bin/env bash
# Phase 1 client bootstrap — checks env, models, starts Match Review.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
PY="${REVIEW_PYTHON:-$HOME/.venvs/soccer-rfdetr312/bin/python3}"
PLAYER="$ROOT/models/people_after_100_epochs.pth"
BALL="$ROOT/models/v12_hard_snaps/post_train/checkpoint.pth"
SMOKE_OUT="$ROOT/data/output/full_match_2min/P10-002/frame_data.csv"
SMOKE_PROC="$ROOT/data/processed/full_match_2min/P10-002/frame_data.csv"

fail() { echo "ERROR: $*" >&2; exit 1; }

[[ -x "$PY" ]] || fail "venv not found: $PY — see docs/product/CLIENT_HANDOVER_QUICKSTART.md"
[[ -f "$PLAYER" ]] || fail "missing $PLAYER — see models/README.md"
[[ -f "$BALL" ]] || fail "missing $BALL — see models/README.md"

# Bundled smoke lives in git under data/processed/; dashboard expects data/output/.
if [[ ! -f "$SMOKE_OUT" ]] && [[ -f "$SMOKE_PROC" ]]; then
  mkdir -p "$ROOT/data/output"
  ln -sfn ../processed/full_match_2min "$ROOT/data/output/full_match_2min"
  echo "Linked data/output/full_match_2min → data/processed/full_match_2min"
fi
[[ -f "$SMOKE_OUT" ]] || fail "missing smoke output — git pull (need data/processed/full_match_2min)"

cd "$ROOT"
export PYTHONPATH=.
export MPLCONFIGDIR=/tmp/mpl-soccer
export USE_TF=0
export TRANSFORMERS_NO_TF=1

echo "Checking PyTorch GPU…"
"$PY" -c "
import torch
ok = torch.cuda.is_available()
print('cuda:', ok, torch.cuda.get_device_name(0) if ok else 'cpu-only')
if not ok:
    print('WARN: GPU not visible — review will be slow; RTX 5090 needs cu128 nightly torch')
" || fail "PyTorch check failed"

bash "$ROOT/scripts/start_review_dashboard.sh" start-bg
echo ""
echo "Match Review → http://127.0.0.1:${REVIEW_PORT:-8501}/"
echo "Smoke data: data/output/full_match_2min/P10-002"
echo "Guide: docs/product/CLIENT_HANDOVER_QUICKSTART.md"
