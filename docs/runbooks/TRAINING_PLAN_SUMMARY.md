# Training Plan Summary: SoccerTrack_sub

## Quick Start

### 1. Prepare Dataset
```bash
cd /workspace/soccer_coach_cv

# Extract frames and generate pseudo-labels
python scripts/prepare_soccertrack_dataset.py \
    --data-dir data/raw/SoccerTrack_sub \
    --output-dir datasets/soccertrack \
    --frame-interval 30 \
    --pseudo-label \
    --confidence-threshold 0.1
```

### 2. Review & Refine Annotations (Optional but Recommended)
- Review pseudo-labels in CVAT or similar tool
- Manually correct ~200-500 "hard" frames
- Focus on frames with ball detections

### 3. Train Model

**Phase 1: Head Training (10 epochs)**
```bash
python scripts/train_detr.py \
    --config configs/training_soccertrack_phase1.yaml \
    --train-dir datasets/soccertrack/train \
    --val-dir datasets/soccertrack/val \
    --output-dir models/soccertrack_training
```

**Phase 2: Full Fine-Tuning (40 epochs)**
```bash
python scripts/train_detr.py \
    --config configs/training_soccertrack_phase2.yaml \
    --train-dir datasets/soccertrack/train \
    --val-dir datasets/soccertrack/val \
    --output-dir models/soccertrack_training \
    --resume models/soccertrack_training/checkpoints/checkpoint_epoch_10_lightweight.pth
```

### 4. Monitor Training
```bash
# MLflow UI
./scripts/start_mlflow_ui.sh

# TensorBoard
tensorboard --logdir logs
```

## Key Optimizations from Strategic Report

✅ **SAHI Integration**: Slicing Aided Hyper Inference for small ball detection  
✅ **Progressive Training**: Freeze backbone → Full fine-tuning  
✅ **Advanced Augmentations**: Motion blur, ISO noise, JPEG compression, MixUp, Mosaic  
✅ **Class Imbalance**: Focal Loss + Weighted Random Sampling  
✅ **Mixed Precision**: FP16/FP32 for training, FP16 for inference  
✅ **TF32 Acceleration**: Enabled for Ampere+ GPUs  

## Expected Results

- **Ball Recall**: >80% (vs baseline 25%)
- **Ball mAP@0.5**: >0.40 (vs baseline ~0.15)
- **Player mAP@0.5**: >0.70
- **Training Time**: 10-20 hours (depending on dataset size)

## Full Documentation

See `TRAINING_PLAN_SOCCERTRACK.md` for complete details.
