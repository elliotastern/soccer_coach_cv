# Team Assignment Status

## Current Issue

All detections in `data/output/37a_20frames/frame_data.json` have `team_id: -1` (unassigned).

## Root Cause

The team clusterer was **not trained** during the 20-frame run. From the log:
```
Team clusterer trained: False
```

## Why Team Assignment Failed

1. **Golden Batch Collection**: The pipeline collects high-confidence (>0.8) player crops during the first `golden_batch_size` frames (default: 500 frames).

2. **Training Requirement**: The team clusterer needs at least **20 crops** to train (see `scripts/process_video_pipeline.py:186`).

3. **20-Frame Run**: With only 20 frames processed:
   - May not have collected 20 high-confidence detections
   - Or detections had confidence < 0.8 threshold
   - Or crop extraction failed

4. **Result**: `team_clusterer_trained = False`, so all players get `team_id = -1` (unassigned).

## Solution

### Option 1: Process More Frames (Recommended)

Run the pipeline with more frames to collect enough crops:

```bash
python scripts/process_video_pipeline.py \
    data/raw/37CAE053-841F-4851-956E-CBF17A51C506.mp4 \
    --model models/checkpoint_best_total_after_100_epochs.pth \
    --output-dir data/output/37a_full \
    --max-frames 500 \
    --golden-batch 100  # Lower from default 500 to train earlier
```

### Option 2: Lower Confidence Threshold

If you have 20+ detections but they're below 0.8 confidence, lower the threshold in `initialize_team_clustering()`:

```python
high_conf_detections = [d for d in detections if d.confidence > 0.6]  # Lower from 0.8
```

### Option 3: Manual Team Assignment

For validation/testing, you can manually assign teams based on jersey colors or positions in the viewer.

## Code Location

- Team clusterer initialization: `scripts/process_video_pipeline.py:148-205`
- Team assignment: `scripts/process_video_pipeline.py:309-314`
- Team clusterer class: `src/logic/team_id.py:TeamClusterer`

## Verification

Check if team clusterer trained:
```bash
grep -i "team.*train" data/output/*/run.log
```

Or in Python:
```python
# After running pipeline
print(f"Team clusterer trained: {pipeline.team_clusterer_trained}")
```
