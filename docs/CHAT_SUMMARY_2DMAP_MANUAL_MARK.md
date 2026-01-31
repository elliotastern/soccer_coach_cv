# Chat Summary: 2D Map Manual Mark & Report (restart tomorrow)

Use this to pick up where we left off.

---

## What We Did

### 1. Marking UI / manual_marks.json correctness
- **Problem:** With Center + 2:TR placed and 1:TL, 3:BR, 4:BL skipped, the report showed blue dots for both TL and TR. Only TR was actually placed.
- **Fix:** Added `_marked_corner_indices_from_points(points)` so blue dots are drawn only for corners that are **actually placed** (not inferred/skipped). When the frame-boundary quad is used (1–2 distinct corners), we no longer overwrite `marked_corner_indices` with `[0, 1]`; we keep the list derived from the saved `points`.

### 2. Web-only marking (no display)
- **Problem:** Running with `--mark` tried the OpenCV window first and crashed (no Qt/display).
- **Fix:** Added `--web` so you can run `python scripts/test_2dmap_manual_mark.py --mark --web` and use only the browser at `http://localhost:5006/mark_ui.html`.

### 3. 2D diagram improvements (R-003 / R001-Person style)
- **Pitch size:** Default map is **1050×680 px** (105 m × 68 m at 10 px/m). Overridable by `--calibration` JSON `map_size`.
- **Player positions:** Map dots use **feet** (bottom-center of bbox): `foot_x = (x_min+x_max)/2`, `foot_y = y_max`. We transform that point with the homography and draw on the map (no longer box center).
- **Pitch drawing:** If `assets/pitch_template.png` exists and is 1050×680, it’s used. Otherwise we draw: green rectangle, center line, center circle (9.15 m), penalty areas (16.5 m deep, 40.32 m wide).

### 4. Regenerate report and verify
- **Regenerate:** `cd soccer_coach_cv && python scripts/test_2dmap_manual_mark.py --use-saved` (uses existing `data/output/2dmap_manual_mark/manual_marks.json`, no re-marking).
- **Verify:** `python scripts/verify_2dmap_report.py` checks: marks file has points + 4 src_corners; `frame_0_map.jpg` size matches calibration or 1050×680; HTML and frame images exist.

### 5. GitHub
- **Committed locally:** 2D map changes + `verify_2dmap_report.py` are committed. Push failed here (no GitHub auth).
- **You:** Run `git push origin main` from your machine (where you’re logged into GitHub).

---

## Key Files

| File | Role |
|------|------|
| `scripts/test_2dmap_manual_mark.py` | Main script: mark UI (or load marks), build homography, generate HTML report with frame + 2D map per frame. |
| `data/output/2dmap_manual_mark/manual_marks.json` | Saved marks: `points` (corner1..4, center; `image_xy`, `inferred`) and `src_corners_xy` (4 corners). |
| `data/output/homography_calibration.json` | Optional: `map_size`, `k_value`, `alpha`. If present and has `map_size`, it overrides default 1050×680. |
| `scripts/verify_2dmap_report.py` | Sanity checks after regeneration: marks shape, map image size, output files. |
| `assets/pitch_template.png` | Optional: if 1050×680, used as 2D diagram background. |

---

## Commands to Restart Tomorrow

```bash
# From project root or soccer_coach_cv/
cd soccer_coach_cv

# Re-mark in browser (if you want to redo marks)
python scripts/test_2dmap_manual_mark.py --mark --web
# Then open http://localhost:5006/mark_ui.html, place/skip corners + center, Save positions.

# Regenerate report from existing marks (no re-marking)
python scripts/test_2dmap_manual_mark.py --use-saved

# Verify output
python scripts/verify_2dmap_report.py

# Push to GitHub (from your machine)
git push origin main
```

---

## Output Locations

- **Report:** `data/output/2dmap_manual_mark/test_2dmap_manual_mark.html`
- **Frames:** `data/output/2dmap_manual_mark/frames/frame_*_picture.jpg`, `frame_*_map.jpg`
- **Marks:** `data/output/2dmap_manual_mark/manual_marks.json`

Open the HTML in a browser: left column = frame with bboxes + center (red) + marked corners (blue); right column = 2D pitch with feet dots (yellow), center (red), marked corners (blue). With your current marks (Center + 2:TR only), only one blue dot (TR) and the red center.
