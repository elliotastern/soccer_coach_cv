# Handoff: 2D Map Half-Pitch (Option A/B) and Verification

**Date:** 2026-02-04  
**Summary:** So you can continue tomorrow from the same place.

---

## Push to GitHub

The latest commit (`48890b0`) is only local. Push from your machine (with GitHub auth):

```bash
cd /path/to/soccer_coach_cv
git push origin main
```

If you use SSH, confirm `git remote -v` then run `git push origin main`. After a successful push, local and GitHub match.

**Note:** You may have other uncommitted changes (e.g. `docs/VIEWING_2DMAP_REPORT.md`, `serve_viewer.py`, `tune_distortion_web.py`, untracked docs/scripts). Push first; commit those separately if needed.

**Continue tomorrow:** Use this doc as the main reference. Sections below: What Was Done, How to Run, Key Files, Possible Next Steps.

---

## What Was Done

### 1. Option A: Half-pitch map (already in place)
- When the quad is built from **TL, TR and the halfway line** (`_build_quad_from_tl_tr_and_halfway`), the map uses **52.5 m × 68 m** (525×680 px).
- Diagram shows one goal (left), right edge = halfway line; players spread across the half-pitch width.
- Flag: `use_half_pitch = True` when quad comes from halfway; `w_map, h_map = HALF_PITCH_MAP_W, HALF_PITCH_MAP_H` (525, 680).

### 2. Option B: Full-pitch diagram with visible half in left
- **CLI:** `--half-pitch-style half_only` (Option A, default) or `--half-pitch-style left_half` (Option B).
- **Option B:** Diagram is full 1050×680 (two goals, center line in middle). Homography still maps to 525×680; players and landmarks are drawn in the **left 525 px** only. Right goal is drawn and emphasized (thicker line).
- In `test_2dmap_manual_mark.py`: `diagram_w`, `diagram_h`, `diagram_center_map_xy`, `half_pitch_diagram` control diagram size/content; report loop uses these and passes `w_map=525, h_map=680` to drawing so content stays in left half.

### 3. Verification script (`scripts/verify_2dmap_report.py`)
- **Usage:** `python scripts/verify_2dmap_report.py [--half-pitch-style half_only|left_half]`
- **Checks:**
  - `manual_marks.json` exists and has points (and src_corners if present).
  - Map image size: 575×730 (half_only) or 1100×730 (left_half).
  - Right goal visible (for left_half): right margin of map has sufficient brightness (goal net).
  - **Accuracy:** Pitch looks green; enough landmark/player-like pixels in content area; **spread** (dots not all in one corner—min 15% span in x and y).
  - **Players match picture:** Loads `frame_0_picture.jpg`, finds green bbox pixels and their mean x; compares to mean x of dots on map; fails if they’re on opposite sides of the pitch (wrong side).
- Exits 0 if all pass, 1 otherwise.

### 4. Report script flags (`scripts/test_2dmap_manual_mark.py`)
- **`--verify`:** After generating the report, runs `verify_2dmap_report.py` with the same `--half-pitch-style` and exits with its code.
- **`--until-valid`:** Runs report, then verify in a loop (max 10 attempts). If verify fails, re-runs the report and verifies again until pass or 10 tries.

---

## How to Run

- **Generate report (Option B) and verify once:**
  ```bash
  python scripts/test_2dmap_manual_mark.py --use-saved data/raw/E806151B-8C90-41E3-AFD1-1F171968A0D9.mp4 --half-pitch-style left_half --verify
  ```

- **Loop until verification passes (report + verify until valid):**
  ```bash
  python scripts/test_2dmap_manual_mark.py --use-saved data/raw/E806151B-8C90-41E3-AFD1-1F171968A0D9.mp4 --half-pitch-style left_half --until-valid
  ```

- **Verify only** (after report already generated):
  ```bash
  python scripts/verify_2dmap_report.py --half-pitch-style left_half
  ```

- **Report output:** `data/output/2dmap_manual_mark/test_2dmap_manual_mark.html`  
  Open via http://localhost:8080/data/output/2dmap_manual_mark/test_2dmap_manual_mark.html or http://localhost:8080/2dmap

---

## Key Files

| File | Role |
|------|------|
| `scripts/test_2dmap_manual_mark.py` | Main 2D map report: half-pitch logic, Option A/B, `--verify`, `--until-valid` |
| `scripts/verify_2dmap_report.py` | Programmatic checks: size, right goal, accuracy (green, spread), players vs picture |
| `data/output/2dmap_manual_mark/manual_marks.json` | Saved marks (TL, TR, halfway, etc.); video path optional |
| `docs/2DMAP_CURRENT_PROBLEM_DETAILED.md` | Problem description (narrow quad, etc.) |
| Plan: `option_a_half-pitch_map_cb8cbe2e.plan.md`, `option_b_full-pitch_left_half_27b3f49c.plan.md` | Option A/B implementation plans |

---

## Possible Next Steps

- **If “players match picture” is too strict:** Relax or tune thresholds in `check_players_match_picture()` (e.g. allow opposite side if difference &lt; 0.5, or skip when green pixels in frame are few).
- **If spread check is too strict:** Lower `min_span_ratio` (e.g. 0.10) in `check_2d_map_accuracy()`.
- **More accuracy checks:** e.g. require center circle visible, or compare multiple frames’ map vs picture.
- **CI:** Run `--use-saved ... --half-pitch-style left_half --verify` in CI and fail the job if exit code is non-zero.

---

## Constants (in code)

- `DEFAULT_MAP_W, DEFAULT_MAP_H = 1050, 680`
- `HALF_PITCH_MAP_W, HALF_PITCH_MAP_H = 525, 680`
- `DIAGRAM_MARGIN = 25`
- Verifier: `EXPECTED_HALF_ONLY_SIZE = (575, 730)`, `EXPECTED_FULL_SIZE = (1100, 730)`
