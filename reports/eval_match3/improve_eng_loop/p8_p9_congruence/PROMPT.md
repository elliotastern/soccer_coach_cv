# P8 ↔ P9 north-end congruence — eng-loop prompt

**Mode:** Loop engineering (agent + scripts). Human growth labeling only where noted.  
**Baseline git:** `749eaae` — *before changing cameras*  
**Entrypoint score:** `python3 scripts/eng_loop_p8_p9_congruence.py` → all gates ≥ **9/10**

---

## Problem (coach observation)

At the **north goal (P6 end)**:

| File | Filename id | Coach sees in still |
|------|-------------|---------------------|
| `p8-005.mp4` | **P8** | Camera / view is to the **left** of the north goal in the frame |
| `P9-004.mp4` | **P9** | Mirror: camera on the **right** of the north goal |

Product docs + diagram today:

| Quadrant | Assigned cam |
|----------|--------------|
| North-left (+y) | **P8** |
| North-right (−y) | **P9** |

**Hypothesis:** capture titles and product wiring treat `p8-005` as north-left and `P9-004` as north-right, but the **physical mounts are swapped**. Landmark clicks used pitch left/right names that do not match what appears on the **photo** (dashboard warns: left/right = from P1 looking north, not photo left/right).

**Goal:** One congruent story everywhere — diagram chips, landmark orders, calibs/H, mosaic cells, fuse labels, pitch panel dots, docs — without breaking Match 3 rules (Pitch 1 meters, defish, emit ≥ 0.80, filename parser).

---

## Strategy (locked for this loop)

Do **all** of the following as one atomic change set (not piecemeal):

### 1. Physical placement (after swap)

| Filename | Product id (unchanged) | True quadrant |
|----------|------------------------|---------------|
| `p8-005.mp4` | **P8** | **North-right** |
| `P9-004.mp4` | **P9** | **North-left** |

**Do not** rename MP4s. **Do not** swap P8 with P10 or P9 with P7.

### 2. Landmark dashboard — mirror touchline (near↔far) click orders

For **P8** and **P9** only, rebuild assignments by swapping pitch landmark ids **left touchline ↔ right touchline** (not `left_*`↔`right_*` north/south families):

| Pitch id pattern | Mirror to |
|------------------|-----------|
| `*_near_*` / `*_near` | `*_far_*` / `*_far` |
| `*_far_*` / `*_far` | `*_near_*` / `*_near` |
| `halfway_near_touch` | `halfway_far_touch` |
| `halfway_far_touch` | `halfway_near_touch` |

Script: `python3 scripts/gold_set/mirror_p8_p9_lr_landmarks.py` (touchline mirror + refit H).

Keep image pixel clicks; only landmark **names** (pitch y sign) change. Re-click manually if RT fails after mirror.

Serve: `python3 serve_viewer.py` → `/reports/eval_match3/landmark_dashboard/index.html`

### 3. Diagram chips — swap P8 ↔ P9

In `reports/eval_match3/landmark_dashboard/index.html` (`CAM_XY` after `hx/hy` layout):

- Move **P8** chip to north-**right** side (−y)
- Move **P9** chip to north-**left** side (+y)
- Update compass sublabels if needed: LEFT · **P10 · P9**, RIGHT · **P7 · P8** (north pair swapped on sides)

Update `cams.json` labels to match:

- P8 → north-right wording
- P9 → north-left wording

### 4. Calibs / H

After re-click + save:

- `reports/eval_match3/match3_pitch_calib/P8_manual.json`
- `reports/eval_match3/match3_pitch_calib/P9_manual.json`

Refit hull / `H_player` if eng-loop player_map requires it:

```bash
python3 scripts/gold_set/match3_landmarks.py  # if batch refit exists
python3 scripts/gold_set/eng_loop_player_map.py
```

**Gate:** `P8` pitch dots → north-**right** on panel; `P9` → north-**left**. Cross-cam RT stays ≤ 0.15 m if tested.

### 5. Mosaic (cw90 compass — current `main`)

Compass: **Left** top · **Right** bottom · **South** left · **North** right.

**Swap only P8 and P9 cells** (keep P10, P7):

| Cell | Before | After swap |
|------|--------|------------|
| Top-right (left touchline, north) | P8 | **P9** |
| Bottom-right (right touchline, north) | P9 | **P8** |

Target `QUAD_GRID`:

```text
Top:    P10 | P9
Bottom: P7  | P8
```

Rotations unchanged: `QUAD_ROTATE_180 = {P10, P7}`.

Update: `src/review/cam_mosaic.py`, `COACH_CORNER`, `.cursor/rules/match3_camera_layout.mdc`, `docs/product/MATCH3_CAPTURE.md`, eng-loop layout scorers.

### 6. Code wiring (if calib path still keyed by filename id)

Filename id stays P8/P9. Calibs are per-id after re-click — no `match3_xy` remap needed **if** calibs were re-fit for the correct quadrant under the same filename.

If any path still assumes “P8 = north-left”, add explicit placement constants:

```python
# scripts/gold_set/match3_quad_placement.py (new, small)
NORTH_END_QUADRANT = {"P8": "north_right", "P9": "north_left"}
```

Use only for docs/tests/mosaic assertions — not for swapping files.

### 7. Review UI + render

- `src/review/app.py` — stacked mosaic + pitch (`map_orient=cw90`) unchanged except grid swap
- Rerender smoke clip:

```bash
python3 scripts/gold_set/render_phase1_check_mosaic.py \
  --start 3050 --match-sec 1 --stride 4 --out-fps 15 \
  --out-dir reports/eval_match3/improve_eng_loop/player_map/check_15s_s4 \
  --out-file coach_mosaic_pitch_11-12s.mp4
```

### 8. Docs

- `docs/product/MATCH3_CAPTURE.md` — north-end table + mosaic grid + note on capture vs mount verification
- `docs/product/MATCH_REVIEW_HANDOVER.md` — mosaic wording
- `reports/eval_match3/improve_eng_loop/p8_p9_congruence/DEAD_ENDS.md` — log if swap fails gates

---

## Verification checklist (must pass before merge)

### A. Visual still proof (save JPGs in this folder)

| Artifact | Pass when |
|----------|-----------|
| `still_P8_north_goal.jpg` | Goal in frame; camera position consistent with **north-right** placement |
| `still_P9_north_goal.jpg` | Mirror; **north-left** |
| `diagram_chips.jpg` | P9 chip left of north end, P8 chip right (north-up diagram) |
| `mosaic_f2397.jpg` | P9 top-right cell, P8 bottom-right; labels match filenames |

### B. Map congruence (frame 3050)

| Check | Pass when |
|-------|-----------|
| `P8` labels on pitch panel | Dots in north-**right** half (−y) |
| `P9` labels on pitch panel | Dots in north-**left** half (+y) |
| Player map pack | `mapped_frac` not worse than pre-swap; P8/P9 off_pitch not worse |
| Clear-ball strip | P_emit ≥ 0.80 on P8/P9 strips if run |

### C. Eng-loop gates (≥ 9/10 each)

```bash
python3 scripts/eng_loop_run_p8_p9_prompt.py   # mirror + all gates
python3 scripts/eng_loop_p8_p9_congruence.py
python3 scripts/eng_loop_coach_mosaic_match.py
python3 scripts/eng_loop_ball_boxes.py
python3 scripts/gold_set/eng_loop_player_map.py
python3 scripts/eng_loop_streamlit_review.py
```

### D. Regression hard no

- Do **not** change `MIN_SUPPORT`, emit 0.80, or FIFA geometry
- Do **not** swap P10/P7 or end cams
- Do **not** disable defish on product path
- Do **not** delete pre-swap calibs — copy to `match3_pitch_calib/_backup_pre_p8_p9_swap/`

---

## Loop workflow (agent)

1. **Branch** from `main` after `749eaae` (e.g. `fix/p8-p9-north-congruence`).
2. **Backup** `P8_manual.json`, `P9_manual.json`, `cams.json` landmark sections.
3. **Proof** — capture P8/P9 stills; write `proof_notes.md` (1 paragraph each).
4. **Landmarks** — mirror L/R orders; re-click; save maps; refit calibs.
5. **Diagram** — swap `CAM_XY` + `cams.json` labels.
6. **Mosaic code** — swap QUAD_GRID cells; update rules/docs.
7. **Score** — run all eng-loop scripts; fix until ≥ 9/10.
8. **Render** — `coach_mosaic_pitch_11-12s.mp4` + optional `check_15s_s4` full slice.
9. **Human spot-check** — 30 s scrub north-end play; confirm ball + one P8/P9 player agree mosaic ↔ pitch.
10. **Commit** only when all gates pass; message e.g. `P8/P9 north-end congruence: swap quadrants, mirror landmarks, align mosaic`.

---

## Wire map

| Piece | Path |
|-------|------|
| Landmark UI | `reports/eval_match3/landmark_dashboard/` |
| Calibs | `reports/eval_match3/match3_pitch_calib/P8_manual.json`, `P9_manual.json` |
| Map / fuse | `src/mapping/match3_xy.py`, `src/review/multicam_fuse.py` |
| Mosaic | `src/review/cam_mosaic.py` |
| Pitch panel | `src/review/pitch1_panel.py` (`map_orient=cw90`) |
| Render | `scripts/gold_set/render_phase1_check_mosaic.py` |
| This loop | `scripts/eng_loop_p8_p9_congruence.py` |

---

## Done definition

North end is **congruent**: coach can trace any `P8` or `P9` dot from mosaic tile → pitch panel → landmark diagram **without left/right inversion**, and eng-loop scores ≥ 9/10 on congruence + player_map + mosaic + ball_boxes.
