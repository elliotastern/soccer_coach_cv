# Training Resume Summary

## Current Status
- **Current Epoch:** 35
- **Target Epochs:** 40 → 50 → 60 (phased approach)
- **Remaining:** 5 epochs to 40, then evaluate
- **Checkpoint:** `models/checkpoint.pth` (474.6 MB)

## Training Plan (Phased)

### Phase 1: Epoch 35 → 40 (Current)
- **Config:** `configs/resume_20_epochs_low_memory.yaml`
- **Remaining:** 5 epochs
- **Status:** Ready to start

### Phase 2: Evaluate at 40
- **Script:** `python scripts/evaluate_training_progress.py configs/resume_20_epochs_low_memory.yaml`
- **Check:** Metrics improving? Overfitting? Ball detection progress?

### Phase 3: Epoch 40 → 50 (if improving)
- **Config:** `configs/resume_to_50_epochs.yaml`
- **Remaining:** 10 epochs
- **Note:** Update checkpoint path after epoch 40

### Phase 4: Evaluate at 50
- **Script:** `python scripts/evaluate_training_progress.py configs/resume_to_50_epochs.yaml`

### Phase 5: Epoch 50 → 60 (if still improving)
- **Config:** `configs/resume_to_60_epochs.yaml`
- **Remaining:** 10 epochs
- **Note:** Update checkpoint path after epoch 50

## Quick Resume Command
```bash
cd /workspace/soccer_cv_ball
# Phase 1 (35→40):
# Use your training script with: configs/resume_20_epochs_low_memory.yaml

# After epoch 40, evaluate:
python scripts/evaluate_training_progress.py configs/resume_20_epochs_low_memory.yaml

# If improving, continue with Phase 3 config
```

## Model Config
- **Architecture:** RF-DETR Base (DETR)
- **Backbone:** ResNet50
- **Encoder:** dinov2_windowed_small
- **Classes:** 2 (1=player, 2=ball)
- **Hidden Dim:** 256 | **Queries:** 300
- **Layers:** Encoder=6, Decoder=6

## Training Settings
- **Batch Size:** 2 | **Grad Accum:** 20 | **Effective Batch:** 40
- **LR:** 0.0002 | **Encoder LR:** 0.00015 | **Weight Decay:** 0.0001
- **Resolution:** 1120×1120 | **AMP:** Enabled
- **Epochs:** 40 (resume from 35)
- **Multi-scale:** Disabled (memory opt)

## Dataset
- **Train:** 7,096 images, 14,869 annotations
- **Val:** 931 images, 1,024 annotations
- **Path:** `models/ball_detection_combined_optimized/dataset/`
- **Format:** COCO JSON

## Loss & Optimizer
- **Loss Coefs:** Class=1.0, Bbox=5.0, GIoU=2.0
- **Focal Loss:** α=0.25, γ=2.0
- **Optimizer:** AdamW
- **EMA:** Enabled (decay=0.993)

## Config File
`configs/resume_20_epochs_low_memory.yaml`

## Dependencies
✅ All installed (rfdetr>=1.3.0, transformers<5.0.0)
