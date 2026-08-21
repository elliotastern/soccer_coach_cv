# Match 3 capture

Camera **id = video title**. The P-code (or Goal id) in the filename is the camera. Do not remap files because a still looks like another view.

## File map

Folder: `data/raw/Match 3/`

| File | Camera |
|------|--------|
| `P1-006.mp4` | P1 |
| `P6-003.mp4` | P6 |
| `P7-001.mp4` | P7 |
| `p8-005.mp4` | P8 |
| `P9-004.mp4` | P9 |
| `P10-002.mp4` | P10 |
| `P_Goal1-007.mp4` | P_Goal1 |
| `P_Goal2-008.mp4` | P_Goal2 |

`P1-006.mp4` means **P1**. `P9-004.mp4` means **P9**. Never assign `P1-006.mp4` to P9 (or the reverse).

Parser: `scripts/gold_set/raw_cam_id.py` (`cam_id_from_raw_name`, `load_match_raw`). Landmark stills must come from that map (`scripts/gold_set/match3_landmarks.py`). Official pitch is **Pitch 1 / Field 1** ([PITCH1_DIMENSIONS.md](PITCH1_DIMENSIONS.md)): **53.90 × 34.84 m** (south), not FIFA 105×68.

## Physical placement on Pitch 1

Axes: **+x north (P6)**, **−x south (P1)**; **+y left**, **−y right** from P1 looking north.

| Cam | Placement |
|-----|-----------|
| P1 | South end |
| P6 | North end |
| P10 | South–left |
| P7 | South–right |
| P8 | North–left |
| P9 | North–right |
| P_Goal1 / P_Goal2 | At their goals |

Diagram compass: **LEFT = P10 · P8**, **RIGHT = P7 · P9** (`reports/eval_match3/landmark_dashboard/`).

## WHOLE PITCH mosaic (coach UI) — locked

Review mosaic order is **not** free to invent. Locked in `src/review/cam_mosaic.py`:

| Cell | Camera | Rotate |
|------|--------|--------|
| Top left | **P10** | **180°** |
| Top right | **P9** | **180°** |
| Bottom left | **P7** | none |
| Bottom right | **P8** | none |

```
Top:    P10 (180°) | P9 (180°)
Bottom: P7         | P8
```

Reference: `reports/eval_match3/improve_eng_loop/cam_stitch_boxes/coach_layout.jpg`. Cursor rule: `.cursor/rules/match3_camera_layout.mdc`.

## If a FOV looks misplaced

Keep the file on its titled camera. Change **diagram position** or landmark clicks only. **Never** swap files across ids (e.g. do not map `P1-006.mp4` → P9). **Never** “fix” the mosaic by swapping P7–P10 cells without an explicit user change to the locked table above.

## Landmark dashboard

`reports/eval_match3/landmark_dashboard/` (cam chips show the video filename). Viewer: `python3 serve_viewer.py` → `/reports/eval_match3/landmark_dashboard/index.html`.

## Fisheye / lens tag dashboard

Tag which Match 3 cams need undistort and tune k1/k2/p1/p2 before any product A/B:

```bash
python3 scripts/gold_set/match3_fisheye_dashboard.py
python3 serve_viewer.py
```

Open http://127.0.0.1:8080/match3-fisheye — saves `reports/eval_match3/fisheye_dashboard/tags.json` (does not change H until a later undistort pass).

Product candidates (`tag=fisheye`, `use_undistort=true`): **P7** (`k1=-0.30`, `k2=-0.08`, `α=0.80`), **P8** (`k1=-0.30`, `α=0.80`), **P9** (`k1=-0.40`, `α=0.80`), **P10** (`k1=-0.32`, `α=0.80`). Brown `cv2.undistort` only.

Landmarks / H for those cams are on **defished** stills. Product map (`src/mapping/match3_xy.map_ball_box`) undistorts raw detection feet with the same params before H. A/B: `python3 scripts/gold_set/ab_match3_undistort_map.py`.

## Defish impact (measured)

Defish is **required** for P7–P10 and materially improves recall/agree vs mapping raw feet through a defish H. On random 5 gallery (750 frames): emits **189 → 247**, agree **3 → 54**; clear-ball proxy R **0.30 → 0.63** on A/B pack. Does not alone hit PoC **0.80** recall on random packs; M1 P10 strip passes both gates with defish wired.

Full numbers, galleries, and do/don't: **[MATCH3_DEFISH.md](MATCH3_DEFISH.md)**.
