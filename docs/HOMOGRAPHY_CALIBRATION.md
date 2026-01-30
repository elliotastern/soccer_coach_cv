# Homography Calibration Guide

## Overview

After validating the fisheye fix (k value), the next step is to create a 2D top-down map of the pitch using homography transformation.

## Scripts

### 1. `fix_homography.py` - Interactive Calibration

**Purpose:** Click 4 points on the pitch to compute the homography matrix for 2D top-down mapping.

**Features:**
- Applies fisheye fix first (k=-0.32, alpha=0.5)
- Interactive point selection with mouse clicks
- Live preview of 2D top-down map
- Saves calibration to JSON and NumPy files

**Usage:**
```bash
python scripts/fix_homography.py
```

**Instructions:**
1. Script shows the first frame (defished)
2. Click 4 points in a **rectangle pattern**:
   - Best choice: Penalty box corners
   - Order: Top-Left → Top-Right → Bottom-Right → Bottom-Left
3. After 4 clicks, homography is computed automatically
4. Video plays showing:
   - Original (defished)
   - 2D Top-Down Map
   - 2D Map (Lines Only - green filtered)
5. Press 'q' to quit
6. Calibration saved to:
   - `data/output/homography_calibration.json`
   - `data/output/homography_matrix.npy`

**Options:**
```bash
python scripts/fix_homography.py --k -0.32 --alpha 0.5 --map-width 600 --map-height 800
```

### 2. `test_homography.py` - Verification Report

**Purpose:** Generate HTML report showing original vs 2D map for multiple frames.

**Usage:**
```bash
python scripts/test_homography.py
```

**Output:**
- `data/output/homography_test/test_homography.html`
- View at: `http://localhost:8080/data/output/homography_test/test_homography.html`

**Options:**
```bash
python scripts/test_homography.py --homography data/output/homography_calibration.json -n 5
```

## Workflow

1. **Calibrate homography:**
   ```bash
   python scripts/fix_homography.py
   ```
   - Click 4 points on penalty box or field corners
   - Verify the 2D map looks correct in the live preview
   - Press 'q' to save and quit

2. **Verify calibration:**
   ```bash
   python scripts/test_homography.py
   ```
   - Opens HTML report with sample frames
   - Check that 2D maps show proper top-down view
   - Verify lines are clean (no floodlight noise)

3. **If calibration is wrong:**
   - Re-run `fix_homography.py`
   - Click different points (try field corners instead of penalty box)
   - Adjust map size with `--map-width` and `--map-height`

## Tips for Good Calibration

### Point Selection
- **Best:** Penalty box corners (large, visible rectangle)
- **Alternative:** Field corners (if visible)
- **Avoid:** Small rectangles, non-rectangular shapes

### Map Size
- Default: 600x800 pixels
- Larger = more detail but slower processing
- Smaller = faster but less detail
- Aspect ratio should match your clicked rectangle

### Verification
- 2D map should show straight lines (not curved)
- Top-down view should look like a bird's-eye view
- Lines-only view should show clean white lines without noise

## Troubleshooting

**Problem:** 2D map looks distorted
- **Solution:** Re-click points, ensure they form a proper rectangle

**Problem:** Lines are missing in lines-only view
- **Solution:** Check green/white HSV thresholds in `get_green_lines()` function

**Problem:** Map is too small/large
- **Solution:** Adjust `--map-width` and `--map-height` parameters

**Problem:** Calibration file not found
- **Solution:** Run `fix_homography.py` first to create the calibration

## Integration

The saved homography matrix can be used in:
- `scripts/process_video_pipeline.py` (pass `--homography` argument)
- Any script that needs pixel-to-pitch coordinate mapping
- The 2D mapping system (`src/analysis/mapping.py`)
