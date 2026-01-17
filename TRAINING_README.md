# Local RF-DETR Training Guide

This guide explains how to train the RF-DETR object detection model locally for player and ball detection.

## Prerequisites

1. **Dataset**: COCO format dataset in `datasets/train` and `datasets/val`
   - Training: ~1,977 images with annotations
   - Validation: ~1,012 images with annotations
   - Classes: `player` (class 0), `ball` (class 1)

2. **Hardware**: 
   - GPU recommended (CUDA-capable)
   - Minimum 8GB GPU memory for batch_size=4
   - Adjust batch_size in `configs/training.yaml` based on available memory

3. **Dependencies**: Install training dependencies
   ```bash
   pip install -r requirements.txt
   ```

## Training Steps

### 1. Prepare Dataset

Ensure your dataset is in COCO format:
```
datasets/
├── train/
│   ├── images/
│   └── annotations/
│       └── annotations.json
└── val/
    ├── images/
    └── annotations/
        └── annotations.json
```

### 2. Configure Training

Edit `configs/training.yaml` to adjust:
- `batch_size`: Based on GPU memory (default: 4)
- `num_epochs`: Training epochs (default: 50)
- `learning_rate`: Initial learning rate (default: 1e-4)
- Model architecture settings

### 3. Start Training

```bash
python scripts/train_detr.py \
    --config configs/training.yaml \
    --train-dir datasets/train \
    --val-dir datasets/val \
    --output-dir models
```

### 4. Monitor Training

- TensorBoard logs: `logs/`
  ```bash
  tensorboard --logdir logs
  ```
- Checkpoints: `models/checkpoints/`
  - `checkpoint_epoch_N.pth`: Regular checkpoints
  - `best_model.pth`: Best model based on validation mAP

### 5. Resume Training

To resume from a checkpoint:
```bash
python scripts/train_detr.py \
    --config configs/training.yaml \
    --train-dir datasets/train \
    --val-dir datasets/val \
    --output-dir models \
    --resume models/checkpoints/checkpoint_epoch_10.pth
```

## Model Export

After training, export the model for inference:

```bash
python scripts/export_model.py \
    --checkpoint models/checkpoints/best_model.pth \
    --output models/detr_trained.pth \
    --config configs/training.yaml
```

## Using Trained Model

Update `configs/default.yaml` to use local model:

```yaml
detection:
  use_local_model: true
  local_model_path: "models/detr_trained.pth"
  confidence_threshold: 0.5
```

Then update `main.py` to use `LocalDetector` instead of `Detector`:

```python
from src.perception.local_detector import LocalDetector

detector = LocalDetector(
    model_path=config['detection']['local_model_path'],
    confidence_threshold=config['detection']['confidence_threshold']
)
```

## Training Configuration Details

### Model Architecture
- **Backbone**: ResNet-50
- **Architecture**: DETR (Detection Transformer)
- **Classes**: 2 (player, ball) + background

### Hyperparameters
- **Optimizer**: AdamW
- **Learning Rate**: 1e-4 with cosine annealing
- **Warmup**: 5 epochs
- **Weight Decay**: 1e-4
- **Gradient Clipping**: 0.1

### Data Augmentation
- Random horizontal flip (50% probability)
- Color jitter (brightness, contrast, saturation, hue)
- Resize to 1333px (DETR standard)

### Evaluation
- **Metric**: Mean Average Precision (mAP)
- **IoU Thresholds**: 0.5, 0.75
- Validation every 5 epochs

## Troubleshooting

### Out of Memory
- Reduce `batch_size` in `configs/training.yaml`
- Reduce image size in augmentation config
- Use gradient accumulation (not implemented, but can be added)

### Slow Training
- Ensure GPU is being used (check CUDA availability)
- Increase `num_workers` in dataset config (if CPU allows)
- Use mixed precision training (can be added)

### Poor Performance
- Increase training epochs
- Adjust learning rate
- Add more data augmentation
- Check dataset quality and annotations

## Expected Training Time

- **GPU (RTX 3090)**: ~2-3 hours for 50 epochs
- **GPU (RTX 2080)**: ~4-5 hours for 50 epochs
- **CPU**: Not recommended (very slow)

## Next Steps

After training:
1. Evaluate model on test set
2. Fine-tune hyperparameters if needed
3. Export model for production use
4. Update inference pipeline to use local model
