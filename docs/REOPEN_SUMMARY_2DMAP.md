# Summary for When You Reopen This Project (2D Map Manual Mark)

## What Was Done

The **2D map manual marking report** (`scripts/test_2dmap_manual_mark.py`) was improved to match the "Integrated Methodologies" plan and follow-up fixes.

### 1. TR/BR inference (FIFA + center constraint)

- **Problem:** With only TL and BL marked (and center from halfway line), TR/BR were inferred by reflection and could produce a degenerate quad.
- **Change:** New helper `_infer_tr_br_from_tl_bl_center(tl_xy, bl_xy, center_xy, w_map, h_map)` uses scipy to find TR and BR in image space so the 4-point homography maps the image center to pitch center (525, 340). Used when the quad is degenerate and we have two placed corners (TL, BL) plus center.
- **Helpers:** `_get_two_placed_corners_tl_bl()` returns the two non-(0,0) corners as TL/BL for inference. `_has_duplicate_corners()` rejects quads with near-identical corners.

### 2. Main flow: when to infer vs frame boundary

- If `_count_unique_corners < 3` or `_quad_area < 100`, the script first tries FIFA inference (two placed corners + center). If that fails, it falls back to `_build_quad_from_two_corners_frame_boundary`.
- After that, if any two corners are duplicate (< 1 px), it tries inference again, then frame boundary, then exits with a clear error.

### 3. Optional y-axis calibration

- `_compute_y_axis_scale_from_positions(map_positions_xy, h_map, ...)` computes a y-scale when observed y-range is too small.
- After building H, if the detector is loaded and `n_frames >= 15`, a pre-pass over the first 15 frames collects player map positions and sets `y_axis_scale`. It is passed into `_draw_boxes_and_landmarks_on_map`, which applies the scale to y when drawing player dots.

### 4. Inference in collect_marks_interactive

- When only TL (corner1) and BL (corner4) are placed and corner2/corner3 are inferred, the script uses `_infer_tr_br_from_tl_bl_center` for TR/BR and saves those in `manual_marks.json`.
- Skipping corner 2 is supported (press S at step 1) so you can mark only TL, BL, then corner4, then Center.

### 5. Map and frame drawing

- **All four corners on map:** The 2D diagram always draws blue markers at all four pitch corners (TL, TR, BR, BL).
- **Halfway line on map:** `_draw_boxes_and_landmarks_on_map` takes `halfway_line_xy` and draws projected halfway_left / halfway_right as yellow circles on the map (with y_axis_scale applied).
- **Red center only when marked:** `center_was_marked = any(p.get("role") == "center" for p in points_from_file)` (default True when no points from file). The red center dot is drawn on the frame and map only when `center_was_marked`; when center is derived from the midpoint of the two yellow halfway points, no red dot is drawn (only 2 blue + 2 yellow = 4 dots).

## Key Files

- **Script:** [scripts/test_2dmap_manual_mark.py](scripts/test_2dmap_manual_mark.py)
- **Marks (saved):** `data/output/2dmap_manual_mark/manual_marks.json`
- **Report:** `data/output/2dmap_manual_mark/test_2dmap_manual_mark.html`
- **Video (default):** `data/raw/E806151B-8C90-41E3-AFD1-1F171968A0D9.mp4`

## Commands

```bash
# Regenerate report with saved marks (use video path if manual_marks.json has no video_path)
python scripts/test_2dmap_manual_mark.py --use-saved data/raw/E806151B-8C90-41E3-AFD1-1F171968A0D9.mp4

# Re-mark corners (opens UI)
python scripts/test_2dmap_manual_mark.py --mark

# View report (serve from project root)
# Open: http://localhost:8080/data/output/2dmap_manual_mark/test_2dmap_manual_mark.html
```

## Git

- **Committed locally:** One commit on `main` with message  
  `2D map: FIFA TR/BR inference, y-axis calibration, halfway on map, hide derived center dot`
- **Push to GitHub:** From your machine (with GitHub auth):  
  `git push origin main`

## Current Marks Behavior

- Your saved `manual_marks.json` has corner1, corner4, halfway_left, halfway_right (no explicit "center" point). Center is derived as the midpoint of the two yellow points; the red dot is **not** drawn. You see 4 dots: 2 blue (corners) + 2 yellow (halfway). TR and BR are inferred via FIFA+center and the report builds a valid quad.
