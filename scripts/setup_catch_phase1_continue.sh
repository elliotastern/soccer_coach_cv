#!/usr/bin/env bash
# Catch RTX 5090 — continue Phase 1 after cu128 torch is verified.
# Does NOT re-run full requirements.txt (breaks torch).
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
VENV="${CATCH_VENV:-$HOME/.venvs/soccer-rfdetr312}"
PY="$VENV/bin/python3"
PIP="$VENV/bin/pip"
PLAYER="$ROOT/models/people_after_100_epochs.pth"
BALL="$ROOT/models/v12_hard_snaps/post_train/checkpoint.pth"

fail() { echo "ERROR: $*" >&2; exit 1; }

[[ -x "$PY" ]] || fail "venv not found: $PY"

echo "=== 1/4 PyTorch cu128 ==="
"$PY" -c "
import torch
print('torch', torch.__version__)
print('cuda', torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'cpu')
if not torch.cuda.is_available():
    raise SystemExit('GPU not visible — run scripts/install_torch_rtx5090.sh')
"

echo "=== 2/4 Model weights ==="
[[ -f "$PLAYER" ]] || fail "missing $PLAYER — AnyDesk from developer (see models/README.md)"
[[ -f "$BALL" ]] || fail "missing $BALL — AnyDesk from developer (see models/README.md)"
ls -lh "$PLAYER" "$BALL"

echo "=== 3/4 pip deps (no torch) ==="
grep -vE '^(torch|torchvision)' "$ROOT/requirements.txt" > /tmp/req-no-torch.txt
"$PIP" install -r /tmp/req-no-torch.txt

echo "=== 4/4 Match 4 raw map ==="
cd "$ROOT"
export PYTHONPATH=.
"$PY" -c "from pathlib import Path; from scripts.gold_set.raw_cam_id import load_match_raw; print(load_match_raw(Path('data/raw/Match 3')))"

echo ""
echo "OK — run: bash scripts/bootstrap_phase1_client.sh"
echo "Then batch: bash scripts/run_batch_match4.sh"
