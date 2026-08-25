#!/usr/bin/env bash
# One-time Catch setup: exchange dirs + openssh-server (key auth from Mac via Tailscale).
set -euo pipefail

EXCHANGE="${HOME}/soccer_exchange"
mkdir -p "${EXCHANGE}/from_catch" "${EXCHANGE}/to_catch"
chmod 700 "${EXCHANGE}"

if ! command -v sshd >/dev/null 2>&1; then
  echo "Installing openssh-server…"
  sudo apt-get update -qq
  sudo apt-get install -y openssh-server
fi
sudo systemctl enable --now ssh 2>/dev/null || sudo systemctl enable --now sshd 2>/dev/null || true

mkdir -p "${HOME}/.ssh"
chmod 700 "${HOME}/.ssh"
touch "${HOME}/.ssh/authorized_keys"
chmod 600 "${HOME}/.ssh/authorized_keys"

TS_IP=""
if command -v tailscale >/dev/null 2>&1; then
  TS_IP="$(tailscale ip -4 2>/dev/null | head -1)"
fi

echo ""
echo "=== Catch SSH sync ready ==="
echo "Exchange: ${EXCHANGE}"
echo "  from_catch/  → Mac pulls (renders, exports)"
echo "  to_catch/    → Mac pushes (weights, patches)"
if [[ -n "${TS_IP}" ]]; then
  echo "Tailscale IP: ${TS_IP}"
  echo "Mac ~/.ssh/config HostName: ${TS_IP}"
else
  echo "Tip: install Tailscale (see docs/product/CATCH_MAC_FILE_SYNC.md)"
fi
echo ""
echo "On Mac:"
echo "  ssh-keygen -t ed25519 -f ~/.ssh/id_ed25519_soccer_catch -N \"\" -C soccer-catch-sync"
echo "  cat ~/.ssh/id_ed25519_soccer_catch.pub   # paste into ~/.ssh/authorized_keys on Catch"
echo "  # or if you know catch password: ssh-copy-id -i ~/.ssh/id_ed25519_soccer_catch.pub catch@${TS_IP:-<IP>}"
echo ""
echo "No catch password? On Catch (AnyDesk), paste pub key line into:"
echo "  nano ~/.ssh/authorized_keys"
echo ""
echo "Mac test: ssh -i ~/.ssh/id_ed25519_soccer_catch catch@${TS_IP:-<IP>} 'echo ok'"
