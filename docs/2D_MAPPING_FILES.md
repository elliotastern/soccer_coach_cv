# 2D Mapping Files (Pixel → Pitch)

Files that implement pixel-to-pitch 2D mapping and how to get the **best** mapping.

---

## Core files

| File | Role |
|------|------|
| **`src/analysis/mapping.py`** | **PitchMapper** – applies homography + y_axis_scale to convert pixel → pitch (meters). Used by the pipeline for every frame. |
| **`src/analysis/homography.py`** | **HomographyEstimator**, `estimate_homography_manual`, `estimate_homography_auto_averaged`, `estimate_homography_auto_with_undistorted`. Builds the 3×3 homography from image ↔ pitch correspondences. |
| **`src/analysis/pitch_keypoint_detector.py`** | Detects pitch landmarks (goals, touchlines, center circle) for automatic homography. Used by homography.py. |
| **`src/analysis/pitch_landmarks.py`** | FIFA-standard pitch coordinates (105×68 m). Reference for pitch space. |
| **`src/analysis/y_axis_calibration.py`** | **refine_homography_with_center_circle**, **calibrate_y_axis_from_center_circle**. Uses center circle (9.15 m) and field width to fix y-axis scale. |
| **`src/analysis/undistortion.py`** | Fisheye/lens distortion correction before homography. Used when `correct_distortion=True`. |
| **`scripts/calibrate_homography.py`** | Interactive tool: click 4+ landmarks on a frame, map to pitch coords, save homography JSON. For manual calibration. |

---

## Best mapping (recommended)

**Use the pipeline’s auto path** (no pre-calibration):

1. **`scripts/process_video_pipeline.py`** – already wires the best path:
   - **HomographyEstimator** from `src/analysis/homography.py`
   - **PitchMapper** from `src/analysis/mapping.py`
   - If no `--homography` JSON: auto-initializes from the first ~15 frames via **estimate_averaged** (multi-frame landmark averaging).
   - Enables **correct_distortion** (fisheye correction), **center circle** refinement, and **y_axis_scale** from `y_axis_calibration.py`.
   - Optionally refines y-axis from player positions after calibration.

2. **Optional pre-calibration** (if auto fails or you want manual control):
   - Run **`scripts/calibrate_homography.py`** on a keyframe (click 4+ landmarks, pick pitch points).
   - Save homography JSON.
   - Pass it to the pipeline: `--homography path/to/homography.json`.

So the **best** 2D mapping is the one the pipeline already uses: **homography.py** (HomographyEstimator, multi-frame auto or manual) → **mapping.py** (PitchMapper with y_axis_scale). No code change needed; just run the pipeline with or without `--homography`.

---

## Flow summary

```
Video frames
    → HomographyEstimator.estimate_averaged() or .estimate()  [homography.py]
        → pitch_keypoint_detector (landmarks)
        → undistortion (optional fisheye fix)
        → estimate_homography_manual() (RANSAC)
        → y_axis_calibration.refine_homography_with_center_circle()
        → y_axis_scale
    → PitchMapper.set_homography(H, y_axis_scale)             [mapping.py]
    → frame_data with x_pitch, y_pitch in meters
```

For manual calibration only: **calibrate_homography.py** → JSON → pipeline `--homography`.
