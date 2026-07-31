# Semantic Segmentation Testing Guide

This guide explains how to test the semantic segmentation-based 2D mapping system.

## Quick Test (No Images Needed)

Run the quick test with a synthetic soccer field:

```bash
python scripts/quick_test_segmentation.py
```

This will:
- Create a synthetic soccer field image
- Test segmentation (using color-based fallback)
- Detect lines and keypoints
- Generate visualization images: `test_segmentation_output.jpg` and `test_segmentation_comparison.jpg`

## Test with Real Images

### Basic Test (Color-Based Detection)

Test with a real soccer field image:

```bash
python scripts/test_segmentation_visual.py --image path/to/your/image.jpg
```

This uses the color-based fallback method (no trained model needed).

### Test with Semantic Segmentation

To use a trained segmentation model:

1. First, ensure you have a trained model checkpoint
2. Update `configs/goal_detection.yaml`:
   ```yaml
   line_segmentation:
     enabled: true
     model_path: "models/pitch_line_segmentation.pth"
     model_type: "deeplabv3"
   ```

3. Run the test:
   ```bash
   python scripts/test_segmentation_visual.py --image path/to/your/image.jpg --use_segmentation
   ```

Or specify a custom config:
```bash
python scripts/test_segmentation_visual.py --image path/to/your/image.jpg --config configs/goal_detection.yaml
```

## Test with Video

Test on a specific frame from a video:

```bash
python scripts/test_segmentation_visual.py --video path/to/video.mp4 --frame 100
```

This will extract frame 100 and test on it.

## What the Tests Show

The visualization shows:
- **Left panel**: Original image
- **Middle panel**: Segmentation mask (white = detected lines)
- **Right panel**: Detected lines (green) + keypoints (colored circles)
  - Red circles: Goals
  - Blue circles: Corners
  - Yellow circles: Penalty boxes
  - White circles: Other landmarks

## Training a Model

To train your own segmentation model:

1. Prepare your dataset:
   - Images: RGB images of soccer fields
   - Masks: Binary masks (white = line pixels, black = background)
   - Organize in folders: `train_images/`, `train_masks/`, `val_images/`, `val_masks/`

2. Run training:
   ```bash
   python scripts/train_pitch_line_segmentation.py \
     --train_images data/train_images \
     --train_masks data/train_masks \
     --val_images data/val_images \
     --val_masks data/val_masks \
     --model_type deeplabv3 \
     --epochs 20 \
     --output models/pitch_line_segmentation.pth
   ```

3. Use the trained model as described above.

## Current Status

- ✅ Segmentation module implemented
- ✅ Integration with keypoint detector complete
- ✅ Color-based fallback working (no model needed)
- ⏳ Trained model: Requires training data and training

The system works with color-based detection immediately. For improved results with occluded lines, train a segmentation model.
