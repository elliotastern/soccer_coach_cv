# 2D Mapping Files (Pixel → Pitch)

Official pitch is **Pitch 1 / Field 1** ([PITCH1_DIMENSIONS.md](../product/PITCH1_DIMENSIONS.md): **53.90 × ~34.8 m**). FIFA 105×68 is legacy only.

Files that implement pixel-to-pitch 2D mapping and how to get the **best** mapping.

---

## Core files

| File | Role |
|------|------|
| **`src/mapping/mapping.py`** | **PitchMapper** – applies homography + y_axis_scale to convert pixel → pitch (meters). Used by the pipeline for every frame. |
| **`src/analysis/homography.py`** | **HomographyEstimator**, `estimate_homography_manual`, `estimate_homography_auto_averaged`, `estimate_homography_auto_with_undistorted`. Builds the 3×3 homography from image ↔ pitch correspondences. |
| **`src/analysis/pitch_keypoint_detector.py`** | Detects pitch landmarks (goals, touchlines, center circle) for automatic homography. Used by homography.py. |
| **`src/analysis/pitch_landmarks.py`** | Legacy FIFA 105×68 landmark table. **Not** Pitch 1. |
| **`docs/product/PITCH1_DIMENSIONS.json`** | **Official Pitch 1 / Field 1** measured marks (53.90 × ~34.8 m) + exact meters for each click landmark. |
| **`src/mapping/pitch_bounds.py`** | `in_pitch_bounds` — defaults to Pitch 1 length/width. |
| **`src/mapping/match3_xy.py`** | Match 3 load H, bbox-foot map, hull support, pitch-space ball fuse. |
| **`src/analysis/y_axis_calibration.py`** | Legacy center-circle refine using FIFA r=9.15 m. Do not use for Pitch 1 (r=3.50 m). |
| **`src/analysis/undistortion.py`** | Fisheye/lens distortion correction before homography. Used when `correct_distortion=True`. |
| **`scripts/calibrate_homography.py`** | Interactive tool: click 4+ landmarks on a frame, map to pitch coords, save homography JSON. For manual calibration. |

---

## Best mapping (recommended)

**Use the pipeline’s auto path** (no pre-calibration):

1. **`scripts/process_video_pipeline.py`** – already wires the best path:
   - **HomographyEstimator** from `src/analysis/homography.py`
   - **PitchMapper** from `src/mapping/mapping.py`
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
