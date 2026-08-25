# Client handover — quickstart (Linux + RTX 5090)

**Repo (public, no login needed):** https://github.com/elliotastern/soccer_coach_cv

**Continuing on catch's PC in Cursor?** Read [CATCH_MACHINE_CURSOR_CONTEXT.md](CATCH_MACHINE_CURSOR_CONTEXT.md) first — session status, verified torch versions, Match 4 mapping, next commands.

This guide gets **Match Review** running for Phase 1 acceptance. Full contract map: [PHASE1_CLIENT_HANDOVER.md](PHASE1_CLIENT_HANDOVER.md).

---

## What is in Git vs what you transfer separately

| Item | In public Git? | How client gets it |
|------|----------------|-------------------|
| Source code, calibs, docs | Yes — `git clone` | — |
| **2 min smoke batch output** | Yes — `data/output/full_match_2min/` | Dashboard works without running batch first |
| **Model weights** (~500 MB total) | No (too large) | AnyDesk / Drive / USB — see [models/README.md](../../models/README.md) |
| **Full match MP4s** | No | Client’s own recordings in `data/raw/Match 3/` |
| Mosaic video in review UI | Needs raw MP4s | Copy P10/P7/P8/P9 into `data/raw/Match 3/` for 4-cam tiles |

---

## 1. Clone (on client machine)

```bash
git clone https://github.com/elliotastern/soccer_coach_cv.git
cd soccer_coach_cv
```

No GitHub account required.

---

## 2. System packages (Ubuntu 22.04 example)

```bash
sudo apt update
sudo apt install -y git curl ffmpeg python3.12 python3.12-venv \
  libgl1 libglib2.0-0 build-essential
nvidia-smi   # must show RTX 5090 + driver
```

---

## 3. Python environment

```bash
python3.12 -m venv ~/.venvs/soccer-rfdetr312
source ~/.venvs/soccer-rfdetr312/bin/activate
pip install -U pip wheel
cd soccer_coach_cv
pip install -r requirements.txt
```

### RTX 5090 — PyTorch cu128 (required)

Stable `cu124` does **not** support Blackwell (sm_120). **Pin both** `torch` and `torchvision` — unpinned install backtracks and fails.

```bash
cd ~/soccer_coach_cv
git pull
bash scripts/install_torch_rtx5090.sh
```

Or manual (when nightly dates skew, install torchvision with `--no-deps`):

```bash
pip uninstall -y torch torchvision torchaudio
pip cache purge
pip install --pre torch==2.12.0.dev20260408+cu128 \
  --index-url https://download.pytorch.org/whl/nightly/cu128 --no-cache-dir
pip install --pre torchvision==0.27.0.dev20260407+cu128 \
  --index-url https://download.pytorch.org/whl/nightly/cu128 --no-deps --no-cache-dir
```

Do **not** re-run `pip install -r requirements.txt` after cu128 (caps `torch<=2.8`). Install other deps with:

```bash
grep -vE '^(torch|torchvision)' requirements.txt > /tmp/req-no-torch.txt
pip install -r /tmp/req-no-torch.txt
```

Verify:

```bash
python3 -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

---

## 4. Model weights (developer transfer)

Place exactly:

```text
models/people_after_100_epochs.pth
models/v12_hard_snaps/post_train/checkpoint.pth
```

See [models/README.md](../../models/README.md) for sizes and checksum note.

---

## 5. Start Match Review

```bash
source ~/.venvs/soccer-rfdetr312/bin/activate
cd soccer_coach_cv
bash scripts/bootstrap_phase1_client.sh
```

Or manually:

```bash
export PYTHONPATH=.
bash scripts/start_review_dashboard.sh start-bg
```

Open **on the client PC:** http://127.0.0.1:8501 (Firefox/Chrome — not a remote IDE preview).

First mosaic frame with live detection: **20–40 s** on GPU.

---

## 6. Coach walkthrough (10 min)

1. Sidebar → **Match** → `P10-002` (from bundled smoke output).
2. **Watch & rate** — mosaic + Pitch 1 + events bar (pass, dribble, movement, recovery, shot).
3. **Save this frame** / **Fix events** → persists `labels.json` / `events.json`.

Detailed UI: [MATCH_REVIEW_HANDOVER.md](MATCH_REVIEW_HANDOVER.md).

---

## 7. Phase 1 acceptance (full delivery)

| Step | Command / check |
|------|-----------------|
| Batch 2 full matches | `bash scripts/run_phase1_full_matches.sh` |
| Delivery manifest | `python3 scripts/gold_set/build_phase1_delivery_manifest.py` |
| Review + export | `frame_data.csv`, `events.json` per camera folder |
| 3rd match handover | Client runs batch on new footage with this guide |

Proof videos (no batch): [phase1_proof/manifest.json](../../reports/eval_match3/improve_eng_loop/phase1_proof/manifest.json)

---

## Remote setup via AnyDesk

1. Client shares AnyDesk ID; you connect.
2. Run steps 1–5 on **his** machine.
3. **Models:** HF pull or AnyDesk `.pth` (~500 MB). **Deliverables (videos/CSV):** set up [CATCH_MAC_FILE_SYNC.md](CATCH_MAC_FILE_SYNC.md) once (Tailscale + SSH key).
4. Open `http://127.0.0.1:8501` in **his** browser.

### Credentials on Catch’s PC (locked)

Developer **GitHub** has other important projects — **never** put developer GitHub auth on Catch:

- Public `git clone` / `git pull` only — no `gh auth login`, no developer PAT/SSH, no github.com as the developer.
- Code changes on the **developer Mac** → `git push`; Catch only `git pull`.
- Do not sign Cursor / IDE into the developer account on Catch’s PC.

**Hugging Face** is fine for Phase 1 weights (not other critical projects). Prefer AnyDesk `.pth` transfer. If using HF on Catch, use a **read-only** token for that model repo only, then `unset HF_TOKEN` before leaving. Never paste write tokens into chat or shared logs.

Before leaving Catch’s desk: clear HF token / logouts; disconnect AnyDesk; tmux jobs may keep running without accounts.

### Mac ↔ Catch file sync (after one-time setup)

Catch stages: `~/soccer_exchange/from_catch/` · Mac pulls: `bash scripts/pull_from_catch.sh` → `~/Downloads/soccer_catch_sync/`. No `catch` Linux password if SSH public key pasted via AnyDesk. Details: [CATCH_MAC_FILE_SYNC.md](CATCH_MAC_FILE_SYNC.md).

Keep dashboard alive after disconnect:

```bash
tmux new -s review
source ~/.venvs/soccer-rfdetr312/bin/activate
cd soccer_coach_cv && export PYTHONPATH=.
bash scripts/run_review_dashboard_foreground.sh
# Ctrl+B, D to detach
```

---

## Troubleshooting

| Issue | Fix |
|-------|-----|
| `sm_120 not compatible` | Install PyTorch **nightly cu128** (step 3) |
| Connection failed `:8501` | `bash scripts/start_review_dashboard.sh restart` |
| No match in sidebar | Confirm `data/output/full_match_2min/P10-002/frame_data.csv` exists after clone |
| No mosaic video | Add raw MP4s under `data/raw/Match 3/` (P-code in filename) |
| Models missing | See `models/README.md` — transfer from developer |
