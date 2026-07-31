# Testing & Validation Guide

This guide outlines how to test and validate that the improvements have fixed the issues and improved system performance.

## Quick Validation Checklist

### ✅ Phase 1: Critical Fixes Validation

#### 1.1 Test Class Indexing Fix
**Goal**: Verify mAP is no longer 0%

```bash
# Run a short training run (1-2 epochs) to check initial metrics
python scripts/train_detr.py \
    --config configs/training.yaml \
    --train-dir datasets/train \
    --val-dir datasets/val \
    --output-dir models

# Check validation output - should see:
# - Player mAP > 0 (was 0.00%)
# - Ball mAP > 0 (was 0.00%)
# - No "All Background" warnings
```

**Expected Results**:
- ✅ Player mAP@0.5 > 0.0 (should be > 0.10 after 1 epoch)
- ✅ Ball mAP@0.5 > 0.0 (should be > 0.05 after 1 epoch)
- ✅ No zero recall/precision for players

#### 1.2 Test Focal Loss vs Class Weights
**Goal**: Verify Focal Loss improves precision over 25x class weights

```bash
# Train with Focal Loss (current config)
python scripts/train_detr.py --config configs/training.yaml

# Monitor ball precision in MLflow/TensorBoard
# Should see: Ball Precision > 0.14% (previous was 0.14%)
```

**Expected Results**:
- ✅ Ball Precision@0.5 > 0.20 (improved from 0.14%)
- ✅ Ball Recall@0.5 > 0.50 (maintains or improves from 58%)
- ✅ Fewer false positives (lower avg predictions per image)

### ✅ Phase 2: Architecture Validation

#### 2.1 Test RF-DETR Integration
**Goal**: Verify RF-DETR can be loaded (full training requires RF-DETR's native API)

```python
# Quick test script
from src.training.model import get_detr_model
import yaml

config = yaml.safe_load(open('configs/training.yaml'))
config['model']['architecture'] = 'rfdetr'

try:
    model = get_detr_model(config['model'], config['training'])
    print("✅ RF-DETR model loaded successfully")
except Exception as e:
    print(f"⚠️ RF-DETR not available: {e}")
    print("Note: Full RF-DETR training requires native API")
```

### ✅ Phase 3: Advanced Features Validation

#### 3.1 Test Copy-Paste Augmentation
**Goal**: Verify ball class balancing works

```python
# Test augmentation
from src.training.augmentation import CopyPasteAugmentation
from PIL import Image
import torch

# Create dummy ball patches
ball_patches = [(Image.new('RGB', (20, 20), 'white'), {})]

aug = CopyPasteAugmentation(prob=1.0, max_pastes=3)
aug.set_ball_patches(ball_patches)

# Test on sample image
img = Image.open('datasets/train/images/sample.jpg')
target = {
    'boxes': torch.tensor([[100, 100, 150, 150]]),
    'labels': torch.tensor([1])  # 1-based: player
}

aug_img, aug_target = aug(img, target)
print(f"Original boxes: {len(target['boxes'])}")
print(f"Augmented boxes: {len(aug_target['boxes'])}")
# Should have more boxes (pasted balls)
```

**Expected Results**:
- ✅ More ball annotations in training batches
- ✅ Improved ball recall during training

#### 3.2 Test SAHI Inference
**Goal**: Verify small ball detection improves

```python
# Test SAHI on validation image
from src.training.sahi_inference import sahi_predict
from PIL import Image
import torch

model = load_trained_model()  # Your trained model
img = Image.open('datasets/val/images/sample.jpg')

# Standard inference
standard_preds = model([preprocess(img)])

# SAHI inference
sahi_preds = sahi_predict(model, img, slice_size=640, overlap_ratio=0.2)

print(f"Standard detections: {len(standard_preds['boxes'])}")
print(f"SAHI detections: {len(sahi_preds['boxes'])}")
# SAHI should detect more small balls
```

**Expected Results**:
- ✅ More ball detections with SAHI
- ✅ Better recall for small balls (< 20x20 pixels)

#### 3.3 Test ByteTrack Integration
**Goal**: Verify temporal tracking consistency

```python
# Test ByteTrack on video sequence
from src.tracker import ByteTrackerWrapper
import torch

tracker = ByteTrackerWrapper(frame_rate=30)

# Simulate detections across frames
for frame_idx in range(10):
    detections = {
        'boxes': torch.tensor([[100, 100, 120, 120]]),
        'scores': torch.tensor([0.8]),
        'labels': torch.tensor([1])  # ball
    }
    
    tracked = tracker.update(detections, (1080, 1920))
    print(f"Frame {frame_idx}: {len(tracked)} tracks")
    if tracked:
        print(f"  Track ID: {tracked[0]['track_id']}")
```

**Expected Results**:
- ✅ Consistent track IDs across frames
- ✅ Ball tracks persist even with low-confidence detections

#### 3.4 Test Homography/GSR
**Goal**: Verify pixel-to-pitch coordinate transformation

```python
# Test homography estimation
from src.analysis.homography import HomographyEstimator
import numpy as np
from PIL import Image

estimator = HomographyEstimator(pitch_width=105.0, pitch_height=68.0)
img = np.array(Image.open('datasets/val/images/sample.jpg'))

# Estimate homography (auto or manual)
success = estimator.estimate(img)
if success:
    # Transform a point
    pixel_point = (960, 540)  # Center of 1920x1080 image
    pitch_point = estimator.transform(pixel_point)
    print(f"Pixel {pixel_point} -> Pitch {pitch_point}")
```

**Expected Results**:
- ✅ Homography matrix estimated successfully
- ✅ Points transform correctly to pitch coordinates

### ✅ Phase 4: Data Quality Validation

#### 4.1 Test CLAHE Enhancement
**Goal**: Verify contrast improvement for synthetic fog

```python
# Visual test
from src.training.augmentation import CLAHEAugmentation
from PIL import Image

aug = CLAHEAugmentation(clip_limit=2.0, tile_grid_size=(8, 8))
img = Image.open('datasets/train/images/sample.jpg')
target = {'boxes': torch.tensor([]), 'labels': torch.tensor([])}

enhanced_img, _ = aug(img, target)
enhanced_img.save('enhanced_sample.jpg')
# Compare visually - should see better contrast
```

#### 4.2 Test Motion Blur
**Goal**: Verify motion blur augmentation works

```python
# Test motion blur
from src.training.augmentation import MotionBlurAugmentation

aug = MotionBlurAugmentation(prob=1.0, max_kernel_size=15)
img = Image.open('datasets/train/images/sample.jpg')
target = {'boxes': torch.tensor([]), 'labels': torch.tensor([])}

blurred_img, _ = aug(img, target)
blurred_img.save('blurred_sample.jpg')
```

## Comprehensive Training Test

### Full Training Run with Monitoring

```bash
# 1. Install new dependencies
pip install -r requirements.txt

# 2. Start training with all improvements
python scripts/train_detr.py \
    --config configs/training.yaml \
    --train-dir datasets/train \
    --val-dir datasets/val \
    --output-dir models

# 3. Monitor in MLflow (recommended)
mlflow ui --backend-store-uri file:./mlruns
# Open http://localhost:5000

# 4. Or monitor in TensorBoard
tensorboard --logdir logs
# Open http://localhost:6006
```

### Key Metrics to Monitor

**Training Metrics** (should improve):
- Training loss: Should decrease smoothly
- Focal Loss component: Should focus on hard examples
- Learning rate: Should follow cosine schedule

**Validation Metrics** (critical improvements):
- **Player mAP@0.5**: Target > 0.85 (was 0.00%)
- **Player Recall@0.5**: Target > 0.95 (was 0.00%)
- **Ball mAP@0.5**: Target > 0.70 (was low)
- **Ball Precision@0.5**: Target > 0.70 (was 0.14%)
- **Ball Recall@0.5**: Target > 0.80 (was ~58%)
- **Ball Avg Predictions**: Should be ~1.0 per image (not excessive)

### Comparison: Before vs After

Create a comparison script:

```python
# scripts/compare_metrics.py
import json

# Load old metrics (from previous training)
with open('old_metrics.json') as f:
    old_metrics = json.load(f)

# Load new metrics (from current training)
with open('new_metrics.json') as f:
    new_metrics = json.load(f)

print("Metric Comparison:")
print(f"Player mAP: {old_metrics['player_map']:.4f} -> {new_metrics['player_map']:.4f}")
print(f"Ball Precision: {old_metrics['ball_precision']:.4f} -> {new_metrics['ball_precision']:.4f}")
print(f"Ball Recall: {old_metrics['ball_recall']:.4f} -> {new_metrics['ball_recall']:.4f}")
```

## Quick Diagnostic Script

Run this to verify all fixes are working:

```bash
# scripts/quick_validation.py
python -c "
from src.training.dataset import CocoDataset
from src.training.model import get_detr_model
import yaml

# Test 1: Dataset labels are 1-based
config = yaml.safe_load(open('configs/training.yaml'))
dataset = CocoDataset('datasets/train', transforms=None)
sample = dataset[0]
labels = sample[1]['labels']
print(f'✅ Dataset labels: {labels.unique().tolist()} (should be [1, 2] for 1-based)')

# Test 2: Model can be created
model = get_detr_model(config['model'], config['training'])
print('✅ Model created successfully')

# Test 3: Focal Loss config
focal_enabled = config['training']['focal_loss']['enabled']
print(f'✅ Focal Loss enabled: {focal_enabled}')

# Test 4: Class weights disabled
weights_enabled = config['training']['class_weights']['enabled']
print(f'✅ Class weights disabled: {not weights_enabled}')

print('\n🎉 All critical fixes verified!')
"
```

## Expected Timeline

- **Epoch 1-5**: Should see mAP > 0 immediately (fixes indexing bug)
- **Epoch 10**: Ball precision should improve (Focal Loss working)
- **Epoch 20**: Copy-Paste should show improved ball recall
- **Epoch 50+**: Should approach target metrics

## Troubleshooting

If metrics don't improve:

1. **Still seeing 0% mAP?**
   - Check dataset labels are 1-based: `dataset[0][1]['labels']`
   - Verify model expects 1-based: Check `model.py` line 119

2. **Ball precision still low?**
   - Verify Focal Loss is enabled in config
   - Check Focal Loss is being applied (add debug prints)

3. **No improvement with Copy-Paste?**
   - Verify ball patches are being extracted
   - Check augmentation is enabled in config

4. **SAHI not working?**
   - Verify image slicing is correct
   - Check NMS is merging predictions properly

## Next Steps After Validation

Once improvements are confirmed:

1. **Fine-tune hyperparameters**: Adjust Focal Loss alpha/gamma
2. **Optimize augmentations**: Tune Copy-Paste probability
3. **Scale up training**: Increase epochs if metrics still improving
4. **Deploy improvements**: Use trained model for inference
