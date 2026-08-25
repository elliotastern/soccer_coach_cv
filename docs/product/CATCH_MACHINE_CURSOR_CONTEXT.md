# Catch client machine — Cursor agent context

**Use this doc** when continuing setup on the client PC in a new Cursor chat.  
**Mac developers:** run long GPU jobs **on Catch** via SSH — [CATCH_REMOTE_COMPUTE.md](CATCH_REMOTE_COMPUTE.md) · rule `catch_remote_compute.mdc`.

Paste or `@`-reference: `docs/product/CATCH_MACHINE_CURSOR_CONTEXT.md`

**Client:** catch · **Machine:** Ubuntu 24.04.3 LTS · **GPU:** NVIDIA GeForce RTX 5090 (Blackwell sm_120) · **Driver:** 595.84 · **CUDA (driver):** 13.2  
**Repo:** https://github.com/elliotastern/soccer_coach_cv (public clone — **do not** log into developer GitHub on this PC)  
**Clone path:** `~/soccer_coach_cv`  
**Venv:** `~/.venvs/soccer-rfdetr312` (Python 3.12)  
**Match videos:** `/home/catch/Documents/Matches/Match 4`  
**Raw symlink in repo:** `data/raw/Match 3` → Match 4 folder (legacy folder name; footage is Match 4)

Related guides: [CATCH_REMOTE_COMPUTE.md](CATCH_REMOTE_COMPUTE.md) · [CLIENT_HANDOVER_QUICKSTART.md](CLIENT_HANDOVER_QUICKSTART.md) · [MATCH_REVIEW_HANDOVER.md](MATCH_REVIEW_HANDOVER.md) · [CATCH_MAC_FILE_SYNC.md](CATCH_MAC_FILE_SYNC.md) · [match4_camera_ids.mdc](../cursorrules/match4_camera_ids.mdc)

---

## What this chat / session already completed

| Step | Status | Notes |
|------|--------|-------|
| `git clone` | ✅ Done | `~/soccer_coach_cv` |
| GPU visible (`nvidia-smi`) | ✅ Done | RTX 5090 |
| Match 4 → repo raw link | ✅ Done | `data/raw/Match 3` → `/home/catch/Documents/Matches/Match 4` |
| P-code symlinks in Match 4 folder | ✅ Done | `P1-match4.mp4` … `P_Goal2-match4.mp4` → `cam-*` files |
| `load_match_raw` parser test | ✅ Passed | After commit `d9e1db0` |
| Python venv | ✅ Done | `~/.venvs/soccer-rfdetr312` |
| PyTorch cu128 | ✅ Done | e.g. `torch 2.10.0+cu128` or nightly `2.12.0.dev…+cu128` — verify with `python3 -c "import torch; …"` |
| pip deps (no torch) | ✅ Done | `grep -vE torch requirements.txt` → pip install |
| Model weights | ✅ Done | HF private repo `eeeeeeeeeeeeee3/soccer-coach-phase1-weights` or `pull_phase1_weights_hf.sh` |
| HF auth on Catch | ✅ Done | `hf auth login` → `~/.cache/huggingface/token` (token name `eeeee`, write) |
| Smoke batch for dashboard | ✅ Done | `ln -sfn ../processed/full_match_2min data/output/full_match_2min` |
| Review dashboard | ✅ Done | `bootstrap_phase1_client.sh` → http://127.0.0.1:8501 |
| Smoke video path | ✅ Done | `ln -sf P10-match4.mp4 P10-002.mp4` in raw folder (or auto via `guess_video_for_run` after `git pull`) |
| `tmux` | ✅ Done | `sudo apt install -y tmux` (3.4) — use for **next** batch detach |
| Match 4 5-min batch | ✅ Done | `data/output/match_4_5min/` (quad, 18000 fr/cam) |
| Mosaic render 5 min | ✅ Done | `reports/eval_match3/improve_eng_loop/match4_5min/coach_mosaic_pitch_5min.mp4` |
| Tailscale | ✅ Done | catch `100.113.134.41` · Mac `100.112.17.93` (example IPs) |
| Mac↔Catch file sync | ✅ Done | SSH key `id_ed25519_soccer_catch` · `~/soccer_exchange/` · see [CATCH_MAC_FILE_SYNC.md](CATCH_MAC_FILE_SYNC.md) |
| Match 4 full batch | ⏳ Optional | `run_batch_match4_full_chunked.sh` in tmux `match4_full` (delivery) |

**Do not redo:** venv, torch cu128, HF model download, smoke symlinks, or `hf auth login` unless token revoked.

**Testing default:** 5-min chunked batch + live review — [PHASE1_BATCH_TESTING.md](PHASE1_BATCH_TESTING.md) · rule `phase1_batch_testing.mdc`.

---

## Installed on Catch (AnyDesk session log — 2026-08-24)

Paste this block into a new Cursor chat on Catch so setup is not repeated.

```text
Machine: catch@catch-System-Product-Name · Ubuntu 24.04.3 · RTX 5090 · driver 595.84
Repo: ~/soccer_coach_cv (public git pull only — NO developer GitHub login)
Venv: ~/.venvs/soccer-rfdetr312 (Python 3.12)

System (apt):
  tmux 3.4-1ubuntu0.1
  # plus tmux deps: libevent-core, libutempter0

Python (venv — do NOT pip install -r requirements.txt after torch):
  torch+cu128, torchvision, rfdetr, streamlit, huggingface_hub<1, … (see setup_catch_phase1_continue.sh)

HF:
  hf auth login saved (~/.cache/huggingface/token)
  Weights repo: eeeeeeeeeeeeee3/soccer-coach-phase1-weights (private)
  Scripts: scripts/push_phase1_weights_hf.sh (Mac) · scripts/pull_phase1_weights_hf.sh (Catch)

Repo layout on Catch:
  data/raw/Match 3 → symlink to /home/catch/Documents/Matches/Match 4
  P*-match4.mp4 symlinks in Match 4 folder (locked cam map — see match4_camera_ids.mdc)
  data/output/full_match_2min → symlink to data/processed/full_match_2min (smoke demo)
  Optional: P10-002.mp4 → P10-match4.mp4 for smoke run name

Services:
  Streamlit review: bash scripts/start_review_dashboard.sh → :8501
  Batch Match 4 (5 min, live review): bash scripts/run_batch_match4_5min.sh → data/output/match_4_5min
  Batch Match 4 (full — slow): bash scripts/run_batch_match4.sh → data/output/match_4

Security:
  Developer GitHub credentials NEVER on this PC
  HF write token OK (soccer weights only); saved via hf auth login

File sync (Mac ↔ Catch, verified):
  Tailscale + SSH key only (~/.ssh/id_ed25519_soccer_catch on Mac)
  Catch: ~/soccer_exchange/from_catch/ (stage) · ~/soccer_exchange/to_catch/ (receive)
  Mac: bash scripts/pull_from_catch.sh → ~/Downloads/soccer_catch_sync/
  No catch Linux password needed if pub key in ~/.ssh/authorized_keys (paste via AnyDesk)
  Doc: docs/product/CATCH_MAC_FILE_SYNC.md
```

---

---

## Match 4 batch — recommended: 5 min + review while running

Full-match batch (`run_batch_match4.sh`) processes **all 8 cams × full MP4 length** (~26+ hours on Catch). For Phase 1 handover, use the **5-minute chunked** script instead:

```bash
# Stop a long full batch if still running
pkill -f batch_pipeline || true

# Detached run (survives AnyDesk disconnect)
tmux new -s match4_5min
source ~/.venvs/soccer-rfdetr312/bin/activate
cd ~/soccer_coach_cv && git pull
bash scripts/run_batch_match4_5min.sh
# Ctrl+B then D to detach
```

**What it does**

| Setting | Value |
|---------|-------|
| Duration | **5 min** per cam @ 60 fps (`18000` frames) |
| Cams | **Quad only** (P10, P9, P7, P8) — coach mosaic |
| Config | `configs/batch_rtx5090.yaml` — no enhance_ball/kalman, 600-frame checkpoints |
| Chunks | **1800 frames** (~30 s video) — cumulative merge after each chunk |
| Output | `data/output/match_4_5min/` |

**Review while batch runs:** keep Streamlit on http://127.0.0.1:8501 → **Expert mode** → output root `data/output/match_4_5min`. Refresh after each chunk (~4 min wall on P10-class footage); frame count grows in sidebar.

**Faster single-cam smoke:**

```bash
CAMS=P10-match4 bash scripts/run_batch_match4_5min.sh
```

**Watch progress:**

```bash
tail -f reports/eval_match3/improve_eng_loop/batch_match4_5min_*.log
watch -n 30 'wc -l data/output/match_4_5min/*/frame_data.csv 2>/dev/null'
```

**Rough ETA (5090, quad 5 min):** ~45–90 min total (4 cams × ~12–22 min each at ~8 fps).

---

## Match 4 batch — full run (optional, slow)

`run_batch_match4.sh` runs **full** `batch_pipeline.py` on **8 cameras one after another** (no `--max-frames`). Time ≈ **sum of each MP4 length** × RF-DETR cost per frame.

**Rough RTX 5090 rule:** ~**5–15 processed frames/sec** (4K, player + ball models). Example:

| Video length (per cam) | ~Time per cam | 8 cams total |
|------------------------|---------------|--------------|
| 10 min @ 60 fps | ~1–3 hours | ~8–24 hours |
| 30 min @ 60 fps | ~3–9 hours | ~1–3 days |
| 60 min @ 60 fps | ~6–18 hours | ~2–6 days |

**Check actual file length on Catch:**

```bash
for f in P10-match4 P9-match4 P7-match4 P8-match4; do
  ffprobe -v error -show_entries format=duration -of csv=p=0 \
    "data/raw/Match 3/${f}.mp4" 2>/dev/null | awk -v f="$f" '{printf "%s %.1f min\n", f, $1/60}'
done
```

**Watch progress:**

```bash
tail -f reports/eval_match3/improve_eng_loop/batch_match4_*.log
ls -la data/output/match_4/
watch -n 60 'wc -l data/output/match_4/*/frame_data.csv 2>/dev/null'
```

**After disconnect:** use `tmux` for the **next** run (`tmux new -s match4` … Ctrl+B D). Current run must keep its terminal open unless restarted (checkpoints may resume per cam).

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

## Next steps (after 5-min batch)

1. Confirm output: `ls data/output/match_4_5min/*/frame_data.csv`
2. Review: Expert mode → output root `data/output/match_4_5min`
3. Delivery check: `python3 scripts/gold_set/build_phase1_delivery_manifest.py`
4. Optional full match later: `bash scripts/run_batch_match4.sh` (overnight, all 8 cams)

**One-shot setup (only if starting fresh on a new machine):**

```bash
source ~/.venvs/soccer-rfdetr312/bin/activate
cd ~/soccer_coach_cv && git pull
bash scripts/setup_catch_phase1_continue.sh   # models + deps + raw map check
bash scripts/bootstrap_phase1_client.sh       # dashboard smoke
# in tmux (recommended):
bash scripts/run_batch_match4_5min.sh        # → data/output/match_4_5min (quad, 5 min, live review)
# full match overnight (optional):
# bash scripts/run_batch_match4.sh          # → data/output/match_4
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

### 4. Batch Match 4 (5 min — recommended)

```bash
source ~/.venvs/soccer-rfdetr312/bin/activate
cd ~/soccer_coach_cv && git pull
tmux new -s match4_5min
bash scripts/run_batch_match4_5min.sh   # quad 5 min → data/output/match_4_5min
# Ctrl+B, D to detach
```

Review while running: **Expert mode** → output root `data/output/match_4_5min`.

Single-cam fastest: `CAMS=P10-match4 bash scripts/run_batch_match4_5min.sh`

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
