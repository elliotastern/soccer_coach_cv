# Catch ↔ Mac file sync (set up once)

**Verified workflow** for developer Mac ↔ Catch RTX 5090: Tailscale + SSH key, no AnyDesk file picker, no developer GitHub on Catch.

Cursor rule: `.cursor/rules/catch_mac_file_sync.mdc`

Exchange folders on Catch:

| Path | Direction |
|------|-----------|
| `~/soccer_exchange/from_catch/` | Catch → Mac (videos, CSV, renders) |
| `~/soccer_exchange/to_catch/` | Mac → Catch (weights, patches) |

Related: [CATCH_REMOTE_COMPUTE.md](CATCH_REMOTE_COMPUTE.md) · [CATCH_MACHINE_CURSOR_CONTEXT.md](CATCH_MACHINE_CURSOR_CONTEXT.md) · [catch_client_credentials.mdc](../cursorrules/catch_client_credentials.mdc)

---

## One-time setup

### 1. Tailscale (both machines, same account)

**Mac:** https://tailscale.com/download → sign in → enable network extension  
**Catch:**

```bash
curl -fsSL https://tailscale.com/install.sh | sh
sudo tailscale up
# open printed URL, approve catch-system-product-name
tailscale status
tailscale ip -4
```

Example (IPs change per tailnet):

| Device | Tailscale IP |
|--------|----------------|
| catch-system-product-name | `100.113.134.41` |
| elliots-mac-mini | `100.112.17.93` |

### 2. SSH server on Catch

```bash
sudo apt install -y openssh-server
sudo systemctl enable --now ssh
sudo systemctl status ssh
```

Or: `bash scripts/setup_catch_ssh_sync.sh`

### 3. SSH key on Mac (private key never on Catch)

```bash
ssh-keygen -t ed25519 -f ~/.ssh/id_ed25519_soccer_catch -N "" -C soccer-catch-sync
```

**Authorize key on Catch** — pick one:

**A) No `catch` password (AnyDesk terminal on Catch):**

```bash
mkdir -p ~/.ssh && chmod 700 ~/.ssh
nano ~/.ssh/authorized_keys
# paste full line from Mac: cat ~/.ssh/id_ed25519_soccer_catch.pub
chmod 600 ~/.ssh/authorized_keys
```

**B) If you know `catch` password:**

```bash
ssh-copy-id -i ~/.ssh/id_ed25519_soccer_catch.pub catch@100.113.134.41
```

**Mac `~/.ssh/config`** (copy from `docs/cursorrules/mac_ssh_config_catch.example`):

```sshconfig
Host catch-soccer
    HostName 100.113.134.41
    User catch
    IdentityFile ~/.ssh/id_ed25519_soccer_catch
    IdentitiesOnly yes
```

**Test:**

```bash
ssh -i ~/.ssh/id_ed25519_soccer_catch catch@100.113.134.41 'echo ok'
# → ok (no password)
```

---

## Daily use

### Catch — stage files for Mac

```bash
cp ~/soccer_coach_cv/reports/eval_match3/improve_eng_loop/match4_5min/coach_mosaic_pitch_5min.mp4 \
  ~/soccer_exchange/from_catch/
```

### Mac — pull

```bash
cd soccer_coach_cv
bash scripts/pull_from_catch.sh
# → ~/Downloads/soccer_catch_sync/
open ~/Downloads/soccer_catch_sync/coach_mosaic_pitch_5min.mp4
```

Or manual rsync:

```bash
rsync -avz --progress -e "ssh -i ~/.ssh/id_ed25519_soccer_catch" \
  catch@100.113.134.41:~/soccer_exchange/from_catch/ ~/Downloads/soccer_catch_sync/
```

### Mac — push to Catch

```bash
bash scripts/push_to_catch.sh models/people_after_100_epochs.pth
```

On Catch: `ls ~/soccer_exchange/to_catch/`

---

## Security

| Do | Don’t |
|----|--------|
| Mac-only SSH private key | Private key or GitHub PAT on Catch |
| `public git pull` on Catch | `gh auth login` on Catch |
| Tailscale mesh + SSH key | Expose SSH on public internet |
| Exchange folders only | Paste tokens in chat |

---

## Troubleshooting

| Error | Fix |
|-------|-----|
| `Connection refused` port 22 | `sudo apt install openssh-server && sudo systemctl start ssh` |
| Password prompt | Paste pub key into Catch `authorized_keys` (step 3A) |
| `catch-soccer` not found | Add `~/.ssh/config` or `export CATCH_SSH_TARGET=catch@100.x.x.x` |
| SSH **times out** to `100.113.134.41` | Almost always **Tailscale VPN down on the Mac** (not a bad SSH key). See below. |
| `Failed to load preferences` / `Tailscale is stopped` | Open **Tailscale.app** → Connect / log in. Quit + reopen if stuck. |
| `open Tailscale.app` → executable missing | Reinstall from [tailscale.com/download/mac](https://tailscale.com/download/mac). |

### Tailscale down (2026-09-05 incident)

When Tailscale is stopped, macOS routes Catch’s `100.x` IP over the **LAN** (`en1` → home gateway) instead of the tunnel (`utun*`). Ping/SSH then time out even if Catch’s PC is on.

**Quick check (Terminal.app):**

```bash
bash scripts/diagnose_catch_tailscale.sh
```

Healthy: `route -n get 100.113.134.41` shows **`interface: utun…`** and `ssh … catch@100.113.134.41 'echo ok'` prints `ok`.  
Unhealthy: interface is **`en…`** → open Tailscale and Connect, then re-run the script.

Do **not** rotate the soccer Catch SSH key for timeouts — fix Tailscale first.

---

## Alternatives

| Method | When |
|--------|------|
| **Tailscale + SSH (this doc)** | Default after one-time setup |
| AnyDesk file transfer | Before SSH works |
| `python3 -m http.server` on Catch + `curl` from Mac over Tailscale IP | One-off, no SSH |
| HF weights repo | Models only |
