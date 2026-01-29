# Fisheye Correction: Alpha Parameter Guide

## Problem: Sides Getting Cut Off

When using `alpha=0.0`, the fisheye correction crops to the "valid region" to avoid black edges, but this cuts off the far sides of the frame.

## Solution: Adjust Alpha Parameter

The `alpha` parameter in `getOptimalNewCameraMatrix` controls the trade-off:

- **`alpha=0.0`**: Crop to valid region (no black edges, but loses sides) ← **Current default**
- **`alpha=0.5`**: Compromise (some black edges, shows more sides) ← **Recommended**
- **`alpha=1.0`**: Full frame (may have black edges, but shows everything)

## How to Fix

### Option 1: Interactive Tuning (Recommended)

Run the interactive tool and tune both `k` and `alpha`:

```bash
python scripts/fix_fisheye.py --k -0.32 --alpha 0.5
```

**Controls:**
- `=` / `-`: Adjust k (straightness)
- `a` / `z`: Adjust alpha (show more/less sides)
- `q`: Quit and note final values

### Option 2: Regenerate Test HTML

Regenerate the test HTML with a higher alpha:

```bash
python scripts/test_fisheye.py --k -0.32 --alpha 0.5 -n 5
```

Then open: `http://localhost:8080/data/output/fisheye_test/test_fisheye.html`

### Option 3: Update Default in Code

Edit `scripts/preprocess.py`:
```python
ALPHA = 0.5  # Change from 0.0 to 0.5 or higher
```

## Recommended Values

For the 37a video:
- **k**: `-0.32` (already correct)
- **alpha**: `0.5` to `0.7` (shows more sides, minimal black edges)

## Testing

1. Run interactive tool: `python scripts/fix_fisheye.py --alpha 0.5`
2. Use `a` key to increase alpha until you see more sides
3. Use `z` key to decrease if black edges become too prominent
4. Note the final alpha value
5. Regenerate test HTML with that alpha

## Files to Update

If you find a good alpha value, update:
- `scripts/preprocess.py` → `ALPHA` constant
- `scripts/test_fisheye.py` → default `--alpha` argument
- Any pipeline scripts that use defishing
