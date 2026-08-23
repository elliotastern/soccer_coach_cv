#!/usr/bin/env bash
# RTX 5090 (Blackwell sm_120) — PyTorch cu128 nightly pair. Run inside project venv.
set -euo pipefail
PY="${REVIEW_PYTHON:-python3}"
INDEX="https://download.pytorch.org/whl/nightly/cu128"
# Pin matched pair (pip fails if torch/torchvision resolved separately).
TORCH_VER="${TORCH_CUDA128_VER:-2.12.0.dev20260407}"
TV_VER="${TORCHVISION_CUDA128_VER:-0.27.0.dev20260407+cu128}"

echo "Uninstalling old torch…"
"$PY" -m pip uninstall -y torch torchvision torchaudio 2>/dev/null || true
"$PY" -m pip cache purge

echo "Installing cu128 nightly: torch==${TORCH_VER} torchvision==${TV_VER}"
"$PY" -m pip install --pre \
  "torch==${TORCH_VER}" \
  "torchvision==${TV_VER}" \
  --index-url "$INDEX" \
  --no-cache-dir

"$PY" -c "
import torch
print('torch', torch.__version__)
print('cuda', torch.version.cuda)
print('available', torch.cuda.is_available())
if torch.cuda.is_available():
    print('gpu', torch.cuda.get_device_name(0))
    x = torch.randn(2, 2, device='cuda')
    print('tensor', x.device)
"
