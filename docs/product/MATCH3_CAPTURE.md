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

## If a FOV looks misplaced

Keep the file on its titled camera. Change **diagram position** or landmark clicks only.

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
