# Catch client machine — Cursor agent context

**Use this doc** when continuing setup on the client PC in a new Cursor chat.  
Paste or `@`-reference: `docs/product/CATCH_MACHINE_CURSOR_CONTEXT.md`

**Client:** catch · **Machine:** Ubuntu 24.04.3 LTS · **GPU:** NVIDIA GeForce RTX 5090 (Blackwell sm_120) · **Driver:** 595.84 · **CUDA (driver):** 13.2  
**Repo:** https://github.com/elliotastern/soccer_coach_cv (public clone — **do not** log into developer GitHub on this PC)  
**Clone path:** `~/soccer_coach_cv`  
**Venv:** `~/.venvs/soccer-rfdetr312` (Python 3.12)  
**Match videos:** `/home/catch/Documents/Matches/Match 4`  
**Raw symlink in repo:** `data/raw/Match 3` → Match 4 folder (legacy folder name; footage is Match 4)

Related guides: [CLIENT_HANDOVER_QUICKSTART.md](CLIENT_HANDOVER_QUICKSTART.md) · [MATCH_REVIEW_HANDOVER.md](MATCH_REVIEW_HANDOVER.md) · [match4_camera_ids.mdc](../cursorrules/match4_camera_ids.mdc)

---

## What this chat / session already completed

| Step | Status | Notes |
|------|--------|-------|
| `git clone` | ✅ Done | `~/soccer_coach_cv` |
| GPU visible (`nvidia-smi`) | ✅ Done | RTX 5090 |
| Match 4 → repo raw link | ✅ Done | `data/raw/Match 3` → `/home/catch/Documents/Matches/Match 4` |
| P-code symlinks in Match 4 folder | ✅ Done | `P1-match4.mp4` … `P_Goal2-match4.mp4` → `cam-*` files |
| `load_match_raw` parser test | ✅ Passed | After commit `d9e1db0` (skips bare `cam-N_*.mp4` when P-code symlinks exist) |
| Python venv | ✅ Created | `~/.venvs/soccer-rfdetr312` |
| `pip install -r requirements.txt` | ⚠️ Partial | Ran once; **must not** re-run full file after cu128 torch (caps `torch<=2.8`) |
| PyTorch cu128 nightly (RTX 5090) | ✅ **Verified working** | See versions below |
| Model weights transfer | ❓ Unknown | Check `models/*.pth` — not in git (~500 MB, AnyDesk from developer) |
| `pip install` other deps (no torch) | ❓ Likely next | See commands below |
| Dashboard / batch on Match 4 | ❓ Not confirmed | Depends on models + deps |

---

## Verified PyTorch install (2026-04-08 session)

These versions were confirmed on catch’s machine:

```text
torch 2.12.0.dev20260408+cu128
torchvision 0.27.0.dev20260407+cu128
cuda True NVIDIA GeForce RTX 5090
tensor ok cuda:0
```

**How it was installed** (nightly index dropped `torch==2.12.0.dev20260407`, so torchvision needed `--no-deps`):

```bash
source ~/.venvs/soccer-rfdetr312/bin/activate
cd ~/soccer_coach_cv

pip uninstall -y torch torchvision torchaudio
pip cache purge

pip install --pre torch==2.12.0.dev20260408+cu128 \
  --index-url https://download.pytorch.org/whl/nightly/cu128 --no-cache-dir

pip install --pre torchvision==0.27.0.dev20260407+cu128 \
  --index-url https://download.pytorch.org/whl/nightly/cu128 --no-deps --no-cache-dir
```

**Preferred going forward:** `bash scripts/install_torch_rtx5090.sh` (auto-picks latest cu128 wheels + `--no-deps` for torchvision). Commit `a13cfa9+`.

**Do not:** `pip install -r requirements.txt` after cu128 — it downgrades/breaks torch. Use:

```bash
grep -vE '^(torch|torchvision)' requirements.txt > /tmp/req-no-torch.txt
pip install -r /tmp/req-no-torch.txt
```

Re-verify anytime:

```bash
python3 -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

---

## Match 4 camera mapping (LOCKED — do not swap by FOV)

Physical recorder `cam-N` ≠ P-code. Symlinks already created:

| P-code | Physical cam | Symlink file |
|--------|--------------|--------------|
| P1 | camera 3 | `P1-match4.mp4` |
| P6 | camera 6 | `P6-match4.mp4` |
| P7 | camera 11 | `P7-match4.mp4` |
| P8 | camera 13 | `P8-match4.mp4` |
| P9 | camera 9 | `P9-match4.mp4` |
| P10 | camera 8 | `P10-match4.mp4` |
| P_Goal1 | camera 7 | `P_Goal1-match4.mp4` |
| P_Goal2 | camera 10 | `P_Goal2-match4.mp4` |

Coach mosaic quad (unchanged): Top P10|P9 (180°) · Bottom P7|P8.  
Pitch calibs in repo are **Match 3 P-code keyed** — may need recalib if camera aim changed on Match 4.

Parser check:

```bash
cd ~/soccer_coach_cv && export PYTHONPATH=.
python3 -c "from scripts.gold_set.raw_cam_id import load_match_raw; print(load_match_raw('data/raw/Match 3'))"
```

---

## Next steps (in order)

**One-shot (after `git pull`):**

```bash
source ~/.venvs/soccer-rfdetr312/bin/activate
cd ~/soccer_coach_cv && git pull
bash scripts/setup_catch_phase1_continue.sh   # models + deps + raw map check
bash scripts/bootstrap_phase1_client.sh       # dashboard smoke
# overnight in tmux:
bash scripts/run_batch_match4.sh              # → data/output/match_4
```

### 1. Confirm model weights

```bash
ls -lh ~/soccer_coach_cv/models/people_after_100_epochs.pth
ls -lh ~/soccer_coach_cv/models/v12_hard_snaps/post_train/checkpoint.pth
```

If missing → AnyDesk file transfer from developer only (see `models/README.md`).

| File | Size |
|------|-----:|
| `people_after_100_epochs.pth` | ~128 MB |
| `v12_hard_snaps/post_train/checkpoint.pth` | ~350 MB |

### 2. Finish Python deps (if not done)

```bash
source ~/.venvs/soccer-rfdetr312/bin/activate
bash scripts/setup_catch_phase1_continue.sh
```

Or manual:

```bash
grep -vE '^(torch|torchvision)' ~/soccer_coach_cv/requirements.txt > /tmp/req-no-torch.txt
pip install -r /tmp/req-no-torch.txt
```

### 3. Start Match Review (smoke data bundled in git)

```bash
source ~/.venvs/soccer-rfdetr312/bin/activate
cd ~/soccer_coach_cv
export PYTHONPATH=.
bash scripts/bootstrap_phase1_client.sh
```

Open **on catch’s browser:** http://127.0.0.1:8501  
Demo without batch: sidebar match `P10-002` from `data/output/full_match_2min/`.

Keep alive after AnyDesk disconnect:

```bash
tmux new -s review
source ~/.venvs/soccer-rfdetr312/bin/activate
cd ~/soccer_coach_cv && export PYTHONPATH=.
bash scripts/run_review_dashboard_foreground.sh
# Ctrl+B, D to detach
```

### 4. Batch Match 4

```bash
source ~/.venvs/soccer-rfdetr312/bin/activate
cd ~/soccer_coach_cv
tmux new -s match4
bash scripts/run_batch_match4.sh    # all 8 P*-match4 symlinks → data/output/match_4
# Ctrl+B, D to detach
```

Review in dashboard: **Expert mode** → output root `data/output/match_4`.

---

## Common failures (already hit on this machine)

| Error | Cause | Fix |
|-------|--------|-----|
| `ResolutionImpossible` torch/torchvision | Unpinned nightly install; dates skew | Pin torch; torchvision `--no-deps` or `scripts/install_torch_rtx5090.sh` |
| `No matching distribution torch==2.12.0.dev20260407` | Nightly rolled to 20260408 | Use `torch==2.12.0.dev20260408+cu128` + torchvision `--no-deps` |
| `sm_120 not compatible` | Stable cu124 torch | cu128 nightly only |
| `timm`/`rfdetr` wants torchvision | Installed torch before torchvision | Install torchvision; warnings clear |
| `missing models/...pth` | Weights not transferred | AnyDesk from developer |
| Wrong camera in mosaic | Mapped `cam-N` → `PN` literally | Use Match 4 table above |

---

## Security / ops notes (credentials)

**Protect developer GitHub** — it has other important projects. Catch’s PC must never hold developer GitHub auth.

| Do on Catch | Don’t on Catch |
|-------------|----------------|
| Public `git clone` / `git pull` only (no login) | `gh auth login`, developer PAT, SSH keys, or github.com as developer |
| Code on **developer Mac** → push; Catch only pull | Cursor / IDE signed in as the developer on Catch’s PC |
| Streamlit / batch in tmux | Leave developer accounts logged in when walking away |

**Hugging Face** — OK for this project (weights only; not other critical work). Prefer AnyDesk copy of the two `.pth` files so Catch needs no HF login. If using HF: upload from Mac; on Catch use a **read-only** token for that model repo only, then `unset HF_TOKEN` before leaving. Never paste write tokens into chat or docs; revoke any token that appeared in a terminal log.

**Also:**
- Model weights (~500 MB) via AnyDesk or scoped HF download — not in git.
- Dashboard URL is **localhost on client machine**, not remote IDE preview.
- First mosaic frame with live detection: **20–40 s** on GPU (normal).
- Cursor rule: `.cursor/rules/catch_client_credentials.mdc` (always apply).

---

## Git commits relevant to this handover

| Commit | What |
|--------|------|
| `d9e1db0` | `load_match_raw` skips `cam-N_*.mp4` when P-code symlinks coexist |
| `7e3c645` | `scripts/install_torch_rtx5090.sh` |
| `a13cfa9` | Script auto-detects nightly versions + torchvision `--no-deps` |
| `e272f7e` | Quickstart torch pin docs |
| `12b967a` | Match 4 camera mapping docs |

Run `git pull` before continuing if clone is behind.

---

## Prompt for a new Cursor chat

Copy-paste:

```text
Continue Phase 1 setup on catch's Ubuntu RTX 5090 machine.
Read docs/product/CATCH_MACHINE_CURSOR_CONTEXT.md first.
PyTorch cu128 is already verified (torch 2.12.0.dev20260408+cu128, RTX 5090 cuda ok).
Next: confirm models/*.pth, finish pip deps without torch, bootstrap dashboard, then batch Match 4 symlinks.
Do not re-run full requirements.txt (breaks torch). Match 4 cam mapping is locked — see doc.
```
