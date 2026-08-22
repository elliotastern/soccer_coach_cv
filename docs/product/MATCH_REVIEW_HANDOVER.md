# Match Review dashboard — handover guide

Phase 1 review app for **recorded match videos** (batch processing, not live cameras).  
Entrypoint: `streamlit run apps/review_dashboard.py` · Coach mode is the default UI.

**Contract alignment:** Fiverr Phase 1 scope is “process multiple match files” + Streamlit review — not live RTSP ingest (that is Product Phase 2+).

---

## Two folders, two jobs

| Folder | What goes there | Used by |
|--------|-----------------|--------|
| **Raw cameras** | One `.mp4` per camera (P10, P7, P8, P9, …) | Batch pipeline + mosaic video |
| **Processed output** | One subfolder **per camera** after batch runs | Dashboard (tracks, events, labels) |

The dashboard does **not** open a folder of MP4s directly. It reads **processed runs** and loads **video** from the raw folder for display.

---

## Step 1 — Lay out recorded cameras (raw)

Put all match videos in **one folder**, with the **P-code in the filename** (camera id = video title).

```text
soccer_coach_cv/
  data/raw/Match 3/
    P10-002.mp4
    P7-001.mp4
    p8-005.mp4
    P9-004.mp4
    P1-006.mp4
    P6-003.mp4
    P_Goal1-007.mp4
    P_Goal2-008.mp4
```

**Rules**

- Do **not** assign a file to a different camera because a view “looks wrong” — the P-code in the name is the camera id. See [MATCH3_CAPTURE.md](MATCH3_CAPTURE.md).
- All cams should be from the **same session**, with the **same start time** and **frame rate**, so frame `2400` on P10 is the same moment as frame `2400` on P7.
- One **flat** folder is what the code expects (`data/raw/Match 3/`). If files live elsewhere, copy or symlink them here.

**Pitch calibration** for Match 3 is already in the repo under `reports/eval_match3/match3_pitch_calib/` (no re-click unless cameras move).

---

## Step 2 — Run batch once per camera (build output)

Run the batch pipeline **once per camera file**, all into the **same output root** for that match:

```bash
cd soccer_coach_cv
export PYTHONPATH=.
export MPLCONFIGDIR=/tmp/mpl-soccer

python3 apps/batch_pipeline.py \
  --video "data/raw/Match 3/P10-002.mp4" \
  --output data/output/my_match_1

python3 apps/batch_pipeline.py \
  --video "data/raw/Match 3/P7-001.mp4" \
  --output data/output/my_match_1

# Repeat for P8, P9, P1, P6 (and any other cams you need)
```

**Two full matches (acceptance):**

```bash
bash scripts/run_phase1_full_matches.sh
```

That writes under `data/output/full_match/` (see script for which default cams).

**Output layout** (one subfolder per camera stem):

```text
data/output/my_match_1/
  P10-002/
    frame_data.csv
    events.json
    corrections.json      # after review edits
    labels.json           # per-frame coach QA
    checkpoints/
  P7-001/
    frame_data.csv
    events.json
  P8-005/
    ...
```

You must batch each camera you want in **export/fuse** data. For the **4-quad mosaic**, P10, P9, P7, P8 must exist as MP4s under `data/raw/Match 3/`.

---

## Step 3 — Start the dashboard

```bash
# Most stable on Mac (opens Terminal — survives Cursor close):
bash scripts/start_review_dashboard.sh

# Or macOS LaunchAgent (auto-restart; logs in ~/Library/Logs/soccer-coach-review/):
bash scripts/start_review_dashboard.sh install-launchd

# Background supervisor only (may stop when agent shell exits):
bash scripts/start_review_dashboard.sh start-bg
```

**Open in a normal browser on the same machine:** [http://127.0.0.1:8501](http://127.0.0.1:8501)

Use Safari or Chrome — not only an IDE embedded preview. First frame load can take **20–40 s** (4-cam RF-DETR); keep the tab open while it loads.

---

## Step 4 — Coach session (UI)

### Coach mode (default)

1. Sidebar → **Match** — pick a processed run (e.g. `P10-002`).
2. **Watch & rate** tab:
   - **Left:** four-camera whole-pitch mosaic (Top P10|P9 · Bottom P7|P8).
   - **Right:** mini Pitch 1 map (yellow ball, blue/red teams).
   - **Events bar** under the video — **Pass**, **Dribble**, **Movement**, **Recovery**, **Shot** (from batch `events.json`).
   - **Previous / Next / Play** — moves along `frame_id` on the selected run’s `frame_data.csv`.
3. Answer the quick questions → **Save this frame** (writes `labels.json`).
4. **Fix events** tab — edit type (all five event kinds), drop bad rows → **Save changes** (updates `events.json`).

Toggle **Expert mode** (top) for output paths, checkpoints, and debug overlays.

### What is fused across cameras

| Piece | Source |
|--------|--------|
| Mosaic tiles (P10, P9, P7, P8) | Raw MP4s in `data/raw/Match 3/` at the same `frame_id` |
| Orange boxes + yellow ball (when on) | RF-DETR on mosaic tiles (first load can take 20–40 s) |
| Pitch map (preferred) | Live detections fused across quads |
| Pitch map (fallback) | Merged `frame_data.csv` from sibling folders under the same output root |
| Events table + bar | `events.json` from the **one** run selected in the sidebar (Pass / Dribble / Movement / Recovery / Shot) |

**Timeline + events** follow the **primary** run you pick (usually P10). **Video mosaic** uses **all** quad cams from the raw folder at that frame index.

---

## If camera files are organized differently

### One folder per match, all cams inside

e.g. `Recordings/Match_A/P10.mp4`, `Match_A/P7.mp4`, …

→ Copy or link all files into `data/raw/Match 3/` (flat), then batch each into `data/output/match_a/`.

### One folder per camera

e.g. `Cam_P10/whole_game.mp4`

→ Rename so the **P-code is in the filename** and place under `data/raw/Match 3/`:

`P10-002.mp4`, not `whole_game.mp4` with no P-code.

### Multiple matches

e.g. `data/output/match_1/P10-002/`, `data/output/match_2/P10-002/`

→ In **Expert mode**, set **Output root** to `data/output/match_1` or `match_2`. Swap raw MP4s in `data/raw/Match 3/` (or set the sidebar **Video file** path) to that session’s files.

---

## One-time machine setup

```bash
git clone https://github.com/elliotastern/soccer_coach_cv.git
cd soccer_coach_cv
python3 -m venv ~/.venvs/soccer-rfdetr312
source ~/.venvs/soccer-rfdetr312/bin/activate
pip install -r requirements.txt
```

**Model weights** (required, not all in git):

| Model | Path |
|--------|------|
| Players | `models/people_after_100_epochs.pth` |
| Ball | `models/v12_hard_snaps/post_train/checkpoint.pth` |

**Recommended:** store videos and run heavy processing on a **fast local SSD**. Slow USB external drives can cause read errors during 4-cam scrubbing.

---

## Minimal checklist

```text
□ Clone repo + install requirements
□ Copy model .pth files into models/
□ Put Match 3 MP4s in data/raw/Match 3/ (P-code in each filename)
□ Batch each camera → data/output/<match_name>/<cam_stem>/
□ bash scripts/start_review_dashboard.sh
□ Open http://127.0.0.1:8501
□ Watch & rate + Save this frame
□ Fix events → Save changes
```

---

## Troubleshooting

| Symptom | What to do |
|---------|------------|
| **Connection failed** | Run `bash scripts/start_review_dashboard.sh install-launchd` (best — macOS auto-restart). Or `bash scripts/start_review_dashboard.sh restart` (supervisor loop). Check `bash scripts/start_review_dashboard.sh status`. Open **Safari/Chrome** at `http://127.0.0.1:8501`, not Cursor preview. Logs: `reports/eval_match3/improve_eng_loop/streamlit_review.log` |
| **No match in sidebar** | Run batch first; each run folder needs `events.json` + `frame_data.csv` |
| **Stuck 20–40 s on a frame** | Normal on first load (4-cam detection). Wait, or Expert → turn off boxes |
| **Disk / USB errors** | Move project + videos to internal SSD |
| **Wants live cameras** | Not Phase 1 — record to MP4, then batch → review |

---

## What Phase 1 is not

- **Not** live RTSP / PoE multi-cam ingest in the dashboard (`apps/live_pipeline.py` is Phase 2+).
- **Not** automatic sync if cameras started at different times or frame rates — align recordings before processing.
- **Not** in-app ball box drawing for training gold — use `serve_viewer.py` → `/gold100` or `/match3-m1` for box-level labels.

---

## Related docs

- [PHASE1_CLIENT_HANDOVER.md](PHASE1_CLIENT_HANDOVER.md) — gig scope → repo map + acceptance checklist
- [PHASE1_SCOPE.md](PHASE1_SCOPE.md) — acceptance and batch-first delivery
- [MATCH3_CAPTURE.md](MATCH3_CAPTURE.md) — camera ids and mosaic layout
- [reports/eval_match3/improve_eng_loop/streamlit_review/PROMPT.md](../../reports/eval_match3/improve_eng_loop/streamlit_review/PROMPT.md) — eng-loop scoring for the review app
