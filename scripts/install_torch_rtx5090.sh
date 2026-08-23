#!/usr/bin/env bash
# RTX 5090 (Blackwell sm_120) — PyTorch cu128 nightly. Run inside project venv.
set -euo pipefail
PY="${REVIEW_PYTHON:-python3}"
INDEX="https://download.pytorch.org/whl/nightly/cu128"

fetch_latest() {
  local pkg="$1"
  curl -fsSL "${INDEX}/${pkg}/" \
    | rg -o "${pkg}-[0-9][^\"]+cu128" \
    | sed "s/^${pkg}-//" \
    | sort -u \
    | tail -1
}

# Nightly torch/torchvision dates often skew by one day; torchvision pins exact torch
# in metadata but the older torch wheel may already be dropped from the index.
TORCH_VER="${TORCH_CUDA128_VER:-$(fetch_latest torch)}"
TV_VER="${TORCHVISION_CUDA128_VER:-$(fetch_latest torchvision)}"

if [[ -z "$TORCH_VER" || -z "$TV_VER" ]]; then
  echo "Could not resolve cu128 nightly versions from ${INDEX}" >&2
  exit 1
fi

echo "cu128 nightly: torch==${TORCH_VER}  torchvision==${TV_VER} (torchvision --no-deps)"

"$PY" -m pip uninstall -y torch torchvision torchaudio 2>/dev/null || true
"$PY" -m pip cache purge

"$PY" -m pip install --pre \
  "torch==${TORCH_VER}" \
  --index-url "$INDEX" \
  --no-cache-dir

# Skip torchvision's strict torch== pin when nightly dates don't match.
"$PY" -m pip install --pre \
  "torchvision==${TV_VER}" \
  --index-url "$INDEX" \
  --no-deps \
  --no-cache-dir

"$PY" -c "
import torch, torchvision
print('torch', torch.__version__)
print('torchvision', torchvision.__version__)
print('cuda', torch.version.cuda)
print('available', torch.cuda.is_available())
if torch.cuda.is_available():
    print('gpu', torch.cuda.get_device_name(0))
    x = torch.randn(2, 2, device='cuda')
    print('tensor', x.device)
"
