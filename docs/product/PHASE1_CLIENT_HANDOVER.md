# Phase 1 — client handover (POC)

Maps the **Fiverr Phase 1 gig** to this repository. Technical detail: [PHASE1_SCOPE.md](PHASE1_SCOPE.md) · status: [PHASE1_STATUS.md](PHASE1_STATUS.md).

---

## Gig scope → what you get

| Gig requirement | Delivered in this repo | Notes |
|-----------------|------------------------|--------|
| **Player + ball tracking** | RF-DETR + ByteTrack (`src/perception/`) | Gig says “YOLO”; product uses **RF-DETR** (commercial-safe, same role). |
| **Team A / B** | Color clustering (`src/perception/team.py`, live fuse in review) | Blue / red on Pitch 1 panel; gray when unsure. |
| **Pitch (x, y) mapping** | Pitch 1 homography + multi-cam fuse (`src/mapping/match3_xy.py`) | **53.90 × ~34.8 m** — not FIFA 105×68. |
| **Batch multiple files** | `apps/batch_pipeline.py` + shell runners below | One subfolder per camera under an output root. |
| **Heuristic events** | `src/events/events.py` | **All 5 types** (pass, shot, recovery, dribble, movement) on eng-loop gold **P_emit 1.0**; dribble/movement suppressed in goal band (fusion noise). |
| **Checkpoints** | `checkpoints/checkpoint_frame_*.json` per run | Incremental saves during batch. |
| **Review UI** | `apps/review_dashboard.py` | Coach mode: mosaic + pitch + **events bar**. Guide: [MATCH_REVIEW_HANDOVER.md](MATCH_REVIEW_HANDOVER.md). |
| **CSV / JSON export** | `frame_data.csv`, `events.csv`, `events.json` | Phase 2-ready schema (see below). |

---

## Quick start (your machine)

```bash
cd soccer_coach_cv
python3 -m venv ~/.venvs/soccer-rfdetr312 && source ~/.venvs/soccer-rfdetr312/bin/activate
pip install -r requirements.txt
# Models (not in git): models/people_after_100_epochs.pth
#                       models/v12_hard_snaps/post_train/checkpoint.pth
```

**1. Raw videos** — flat folder, P-code in filename:

`data/raw/Match 3/P10-002.mp4`, `P1-006.mp4`, `P7-001.mp4`, …

**2. Batch** (one command per camera, same output root):

```bash
export PYTHONPATH=.
python3 apps/batch_pipeline.py --video "data/raw/Match 3/P10-002.mp4" --output data/output/full_match
python3 apps/batch_pipeline.py --video "data/raw/Match 3/P1-006.mp4" --output data/output/full_match
```

Or overnight acceptance run:

```bash
bash scripts/run_phase1_full_matches.sh
```

**3. Review dashboard:**

```bash
bash scripts/start_review_dashboard.sh status   # check
bash scripts/start_review_dashboard.sh          # start
# Open Safari/Chrome: http://127.0.0.1:8501
```

**4. Delivery check:**

```bash
python3 scripts/gold_set/build_phase1_delivery_manifest.py
# → reports/eval_match3/improve_eng_loop/delivery_manifest.json
```

---

## Export schema (Phase 2-ready)

**`frame_data.csv`** — per player/ball row per frame:

| Column | Meaning |
|--------|---------|
| `Timestamp` | Seconds |
| `Team_ID` | 0 / 1 / -1 (ball) |
| `Player_ID` | Track id; **-1** = ball |
| `Event` | Row label (`movement` default on tracks) |
| `Location_X`, `Location_Y` | Pitch 1 meters |
| `frame_id` | Source frame index |
| `confidence` | Emit confidence |

**`events.json`** — heuristic emits (pass / shot / recovery today):

```json
{
  "match_id": "P10-002",
  "events": [
    {
      "type": "pass",
      "start_frame": 1958,
      "confidence": 0.85,
      "start_location": { "x": -11.0, "y": 6.7 }
    }
  ]
}
```

---

## Demo assets (no batch required)

| Asset | Path |
|-------|------|
| Phase 1 proof videos (ball / map / events / …) | [phase1_proof/manifest.json](../../reports/eval_match3/improve_eng_loop/phase1_proof/manifest.json) |
| Official check clip (25 s mosaic) | [phase1_check/coach_mosaic_pitch_min.mp4](../../reports/eval_match3/improve_eng_loop/phase1_check/coach_mosaic_pitch_min.mp4) |
| Labelled ball strips (P10 / P8) | `data/processed/gold_sets/match3_quad_p10_31/`, `match3_quad_p8_87/` |
| Ball bbox + best-cam gallery | [quad_pitchmap_gallery_v12_hard/index.html](../../reports/eval_match3/quad_pitchmap_gallery_v12_hard/index.html) |

Rebuild proof pack: `python3 scripts/gold_set/build_phase1_proof_pack.py`

---

## Acceptance checklist (gig §5)

```text
□ Batch P10-002 + P1-006 → data/output/full_match/<cam>/
□ delivery_manifest.json shows acceptance_met: true
□ Review: Watch & rate + Fix events on processed run
□ Handover: you process a 3rd match with this guide
```

**Sample output today (2 min smoke, not full match):**  
`data/output/full_match_2min/P10-002/` — use for UI walkthrough before full batch finishes.

---

## Known gaps (honest)

| Item | Status |
|------|--------|
| 2 **full** matches through batch | Run `run_phase1_full_matches.sh` (hours on GPU) |
| 3rd-match handover session | Manual — use [MATCH_REVIEW_HANDOVER.md](MATCH_REVIEW_HANDOVER.md) |
| Dribble + **movement** heuristic emits | **Done (E2)** — co-movement gate (not goal-band exclusion) |
| Live RTSP / wearables | Phase 2+ — not in Phase 1 scope |

---

## Related docs

- [MATCH_REVIEW_HANDOVER.md](MATCH_REVIEW_HANDOVER.md) — dashboard step-by-step  
- [MATCH3_CAPTURE.md](MATCH3_CAPTURE.md) — camera ids and mosaic layout  
- [PITCH1_DIMENSIONS.md](PITCH1_DIMENSIONS.md) — official pitch meters  
