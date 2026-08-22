# Phase 1 review dashboard — labeling prompt

**App:** `streamlit run apps/review_dashboard.py` → http://127.0.0.1:8501  
**Code:** `src/review/app.py` · entry `apps/review_dashboard.py`  
**Phase 1 pillar:** Safety & review — low-confidence flags, manual corrections persisted to the dataset ([PHASE1_SCOPE.md](../../../../docs/product/PHASE1_SCOPE.md) §3).

Use this prompt in two modes:

| Mode | Who | Outcome |
|------|-----|---------|
| **Growth engineering** | Human (coach / client) | Correct labels, events, and flags that **grow** the gold dataset and exported `events.json` |
| **Loop engineering** | Agent + scripts | Automate verify → fix → re-score until eng-loop **≥ 9/10** |

*(If you meant a different spelling than “growth”, treat **Growth** = human-in-the-loop labeling; **Loop** = `eng_loop_*.py` gates below.)*

---

## Goal

Ship a **local Streamlit coach dashboard** where the client can:

1. Scrub Match 3 (or batch run) video with **locked WHOLE PITCH mosaic** + **Pitch 1** panel.
2. **Label / correct** emitted outputs (events, ball presence, team colors, bad rows).
3. **Persist** corrections back to the run folder (`events.json`, `corrections.json`, future `labels.json`).
4. Pass automated stability gates without LaCie USB EIO crashes.

Precision-first: when unsure, **drop / gray / do nothing** — never invent a ball, team, or event below emit conf **0.80**.

---

## Growth engineering (human labeling)

### What to label in the dashboard

| Label type | Where in UI | Persist to | Done when |
|------------|-------------|------------|-----------|
| **Heuristic events** | Manual corrections table (type, frame, conf, x/y, keep/drop) | `corrections.json` + `events.json` | Client fixes bad pass/shot/recovery rows; `reviewed: true` in metadata |
| **Ball / player verify** | Verify labels — video + pitch + **Frame labels** panel | `labels.json` per run | Orange box on ball; yellow dot on pitch; Save frame label |
| **Team A/B** | Pitch 1 panel (T0 blue / T1 red / gray unassigned) | Visual QA only today | Two kits visible → both colors; unsure stays gray |
| **Low-confidence flags** | Scrub to flagged frames; drop or edit in corrections | Same as events | No published row with conf &lt; 0.80 unless explicitly kept for audit |

### Human workflow (coach session)

1. **Start dashboard**
   ```bash
   cd "/Volumes/LaCie/Projects/Soccer Coach CV"
   PYTHONPATH=. streamlit run apps/review_dashboard.py
   ```
   Or: `bash scripts/start_review_dashboard.sh`

2. **Pick run** — sidebar `Output root` → `data/output/<match_run>/` (must have `frame_data.csv` + `events.json`).

3. **Video path** — sidebar `Video file` → Match 3 raw file for that cam (e.g. `data/raw/Match 3/P10-002.mp4`). Id = filename P-code ([match_raw_camera_ids](../../../../.cursor/rules/match_raw_camera_ids.mdc)).

4. **Locked product toggles** (do not “fix” by turning off):
   - Camera view: **WHOLE PITCH** — Top **P10|P9 (180°)** · Bottom **P7|P8**
   - **Defish P7–P10: ON**
   - **RF-DETR boxes: ON** (first frame 20–40 s on 4 cams — normal)

5. **Scrub** — Prev/Next or slider; use **Only frames with exported ball** to focus ball emits. Keep **Play speed low** (≤ 2 fps) or pause — play + 4-cam detect hammers USB.

6. **Verify per frame**
   - **Video:** orange `BALL` box on visible ball; green player boxes on people.
   - **Pitch:** yellow ball dot when fused emit; team-colored player dots; N-up / S-dn Pitch 1 (**53.90 × ~34.8 m**, not FIFA).

7. **Correct events** — expand **Manual corrections**:
   - Edit `type`, `start_frame`, `confidence`, `x`, `y`.
   - Uncheck `keep` to drop a bad emit.
   - Add review notes → **Persist corrections**.

8. **Detection gold (separate editors)** — box-level train/eval labels still use HTML editors (not Streamlit yet):
   - Gold100: `python3 serve_viewer.py` → http://localhost:8080/gold100
   - Match 3 M1 strips: http://127.0.0.1:8080/match3-m1 ([MATCH3_M1_STRIP.md](../../../../docs/product/MATCH3_M1_STRIP.md))

9. **Handover checklist** — client can process a 3rd match when they can repeat steps 2–7 without agent help ([PHASE1_SCOPE.md](../../../../docs/product/PHASE1_SCOPE.md) §5.2).

### Growth rules (hard)

- **Do not** swap camera files across P-codes because a tile “looks wrong”.
- **Do not** turn defish OFF for product review (A/B only).
- **Do not** lower emit conf below **0.80** to “make more events”.
- **Do not** use FIFA 105×68 — Pitch 1 meters only.
- **Do** write corrections to disk before closing the session.
- **Do** prefer drop/gray over a wrong label.

---

## Loop engineering (automated gates)

### Gate A — HTTP stability

```bash
bash scripts/eng_loop_streamlit_review.sh 8501 10
```

Report: `reports/eval_match3/improve_eng_loop/streamlit_stability_score.json`  
Pass: score **≥ 9.0** (10 consecutive HTTP 200, body &gt; 500 B).

### Gate B — Frame review (no EIO / no crash)

```bash
bash scripts/eng_loop_frame_review_eio.sh 8501 10
```

Report: `reports/eval_match3/improve_eng_loop/frame_review_eio/stability_score.json`  
Pass: score **≥ 9.0** — Streamlit `AppTest` runs with play OFF, dets OFF (LaCie-safe).

### Gate C — Product slice (visual, Match 3 @ frame 2400)

Manual or screenshot script until mean **≥ 9/10** on:

| # | Component | Done when (≥9) |
|---|-----------|----------------|
| 01 | **Import / wire** | `streamlit run apps/review_dashboard.py` loads without ImportError |
| 02 | **Run discovery** | Sidebar lists runs with `events.json` under output root |
| 03 | **Frame CSV** | `frame_data.csv` loads; ball rows `Player_ID == -1` navigable |
| 04 | **Mosaic layout** | WHOLE PITCH = P10\|P9 top 180°, P7\|P8 bottom |
| 05 | **Defish default ON** | `cam_view_defish_on` True; tiles straightened |
| 06 | **RF-DETR default ON** | `show_dets_ball_on` True when paused |
| 07 | **Ball box lock** | Orange box within ~25 px of ball on clear frames |
| 08 | **Pitch ball dot** | Yellow dot when product fuse emits; absent when no emit |
| 09 | **Team colors** | T0/T1 on pitch when two kits; gray when unsure |
| 10 | **Events table** | Summary counts + corrections editor visible when `events.json` exists |
| 11 | **Persist corrections** | Persist writes `corrections.json` + updates `events.json` |
| 12 | **Checkpoint pick** | Sidebar can load `checkpoint_*.json` events |
| 13 | **EIO soft-fail** | USB blip shows warning + pauses play; log `/tmp/scv_frame_review_errors.log` |
| 14 | **Play safe** | Play does not force 4-cam detect every tick |
| 15 | **Pitch 1 panel** | 53.90 m length; goals/boxes from `PITCH1_DIMENSIONS.json` |
| 16 | **MAP-BALL debug off** | Default OFF; optional sidebar debug only |
| 17 | **Hide sidebar** | Top button toggles wide mosaic view |
| 18 | **Static fallback** | `scripts/build_review_partial_html.py` produces offline snapshot |
| 19 | **Gate A pass** | `streamlit_stability_score.json` pass |
| 20 | **Gate B pass** | `frame_review_eio/stability_score.json` pass |

Target eng-loop script: `scripts/eng_loop_streamlit_review.py` → `reports/eval_match3/improve_eng_loop/streamlit_review/scores.json`.

---

## Locked product context

- **Mosaic:** `QUAD_GRID = [["P10","P9"],["P7","P8"]]`, `QUAD_ROTATE_180 = {P10, P9}` (`src/review/cam_mosaic.py`)
- **Defish:** P7–P10 undistort before H; detect after defish on shown pixels ([match3_defish](../../../../.cursor/rules/match3_defish.mdc))
- **Ball fuse:** F1+F2+F0+F3 + hold; emit ≥ 0.80; agree ≤ 4 m ([match3_clear_ball](../../../../.cursor/rules/match3_clear_ball.mdc))
- **Stack:** RF-DETR + ByteTrack in batch; not YOLO

---

## Wire map

| Piece | Path |
|-------|------|
| Streamlit UI | `src/review/app.py` |
| Mosaic | `src/review/cam_mosaic.py` |
| Fuse / map | `src/review/multicam_fuse.py`, `src/mapping/match3_xy.py` |
| Team live | `src/review/team_live.py` |
| Pitch panel | `src/review/pitch1_panel.py` |
| Frame sync / draw | `src/review/frame_sync.py` |
| Start script | `scripts/start_review_dashboard.sh` |
| Loop A | `scripts/eng_loop_streamlit_review.sh` |
| Loop B | `scripts/eng_loop_frame_review_eio.sh` |
| Frame labels | `src/review/frame_labels.py` |
| Eng-loop score | `scripts/eng_loop_streamlit_review.py` |
| Partial HTML | `scripts/build_review_partial_html.py` |

---

## Hard no

- FIFA 105×68 or penalty-spot geometry on Match 3
- Rearrange mosaic by intuition (Far/Near swap)
- Default defish OFF or MAP-BALL ON in coach mode
- Force team 0/1 on every player
- Phase 2 multi-cam temporal fusion in the review UI
- Train from `prelabels/` — gold XML only ([TRAIN_LABEL_SOURCE_OF_TRUTH.md](../../../../docs/ball_detection/TRAIN_LABEL_SOURCE_OF_TRUTH.md))

---

## Done definition

**Growth:** Client completes a labeling session — corrections on disk, can explain ball vs pitch vs events panels, no agent needed for a 3rd match handover.

**Loop:** Gates A + B pass; core components 04–09 and 11 ≥ 9; PHASE1_STATUS review row stays **≥ 7/10** ([PHASE1_STATUS.md](../../../../docs/product/PHASE1_STATUS.md)).

---

## Agent instructions (copy below the line)

You are improving the Phase 1 Streamlit review dashboard for soccer coach labeling.

**Growth path:** Add or polish human labeling UX — event corrections, frame flags, notes, future in-app box edits — always persist to the run folder. Precision-first: conf &lt; 0.80 → drop. Pitch 1 meters only.

**Loop path:** Run `eng_loop_streamlit_review.sh` and `eng_loop_frame_review_eio.sh`; fix crashes/EIO; score visual components at frame 2400; write `streamlit_review/scores.json`. Do not break locked mosaic/defish/fuse.

Read `src/review/app.py` and sibling modules before editing. Match existing eng-loop PROMPTs under `reports/eval_match3/improve_eng_loop/*/PROMPT.md`. Small diffs only.
