#!/usr/bin/env python3
"""
Diagnostic script to investigate validation mAP = 0.0 issue
"""
import torch
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

from src.training.model import get_detr_model
from src.training.dataset import CocoDataset
from src.training.augmentation import get_val_transforms
from src.training.evaluator import Evaluator
from torch.utils.data import DataLoader
from src.training.collate import collate_fn

print("=" * 60)
print("VALIDATION DIAGNOSTIC")
print("=" * 60)

# Load validation dataset
print("\n1. Loading validation dataset...")
val_dataset = CocoDataset('/workspace/datasets/val', transforms=get_val_transforms({'resize': 1333}))
print(f"   Validation samples: {len(val_dataset)}")

# Load model
print("\n2. Loading model from checkpoint...")
checkpoint = torch.load('models/checkpoints/latest_checkpoint.pth', map_location='cpu', weights_only=False)
model = get_detr_model(checkpoint['config']['model'])
model.load_state_dict(checkpoint['model_state_dict'], strict=False)
model.eval()
print("   Model loaded successfully")

# Get a few samples
print("\n3. Testing on sample images...")
val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False, collate_fn=collate_fn)

evaluator = Evaluator({'iou_thresholds': [0.5, 0.75], 'max_detections': 100})

all_predictions = []
all_targets = []

for i, (images, targets) in enumerate(val_loader):
    if i >= 5:  # Test on first 5 images
        break
    
    print(f"\n   Image {i+1}:")
    target = targets[0]
    print(f"     Target: {len(target['boxes'])} boxes, labels: {target['labels'].unique() if len(target['labels']) > 0 else 'none'}")
    
    # Run inference
    with torch.no_grad():
        outputs = model(images)
    
    output = outputs[0]
    print(f"     Prediction: {len(output['boxes'])} boxes, labels: {output['labels'].unique() if len(output['labels']) > 0 else 'none'}")
    print(f"     Scores range: {output['scores'].min():.4f} - {output['scores'].max():.4f}")
    
    # Check label mapping
    if len(output['labels']) > 0:
        unique_labels = output['labels'].unique()
        print(f"     Unique prediction labels: {unique_labels.tolist()}")
        print(f"     Label distribution: {torch.bincount(output['labels'].long())}")
    
    all_predictions.append(output)
    all_targets.append(target)

# Run evaluation
print("\n4. Running evaluation...")
eval_metrics = evaluator.evaluate(all_predictions, all_targets)
print(f"   mAP: {eval_metrics['map']:.4f}")
print(f"   Precision: {eval_metrics['precision']:.4f}")
print(f"   Recall: {eval_metrics['recall']:.4f}")
print(f"   F1: {eval_metrics['f1']:.4f}")

# Check for issues
print("\n5. Diagnostic checks...")
issues = []

# Check if predictions have any non-background labels
total_preds = sum(len(p['labels']) for p in all_predictions)
non_bg_preds = sum(len(p['labels'][p['labels'] > 0]) for p in all_predictions)
if non_bg_preds == 0:
    issues.append("⚠️  All predictions are background (label 0 or negative)")
else:
    print(f"   ✓ Found {non_bg_preds} non-background predictions out of {total_preds} total")

# Check if targets have annotations
total_targets = sum(len(t['boxes']) for t in all_targets)
if total_targets == 0:
    issues.append("⚠️  No target annotations found in validation set")
else:
    print(f"   ✓ Found {total_targets} target annotations")

# Check label ranges
for i, (pred, target) in enumerate(zip(all_predictions, all_targets)):
    if len(pred['labels']) > 0:
        pred_min, pred_max = pred['labels'].min().item(), pred['labels'].max().item()
        if pred_min < 0 or pred_max > 1:
            issues.append(f"⚠️  Image {i+1}: Prediction labels out of range [0,1]: [{pred_min}, {pred_max}]")
    if len(target['labels']) > 0:
        target_min, target_max = target['labels'].min().item(), target['labels'].max().item()
        if target_min < 0 or target_max > 1:
            issues.append(f"⚠️  Image {i+1}: Target labels out of range [0,1]: [{target_min}, {target_max}]")

if issues:
    print("\n   ISSUES FOUND:")
    for issue in issues:
        print(f"   {issue}")
else:
    print("   ✓ No obvious issues found")

print("\n" + "=" * 60)
