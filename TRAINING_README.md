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

#### TensorBoard
TensorBoard logs are saved to `logs/`:
```bash
tensorboard --logdir logs
```

#### MLflow (Recommended)
MLflow provides comprehensive experiment tracking with a web UI. It's enabled by default in `configs/training.yaml`.

**View MLflow UI:**
```bash
# Using helper script
./scripts/start_mlflow_ui.sh

# Or manually
mlflow ui --backend-store-uri file:./mlruns
```

Then open http://localhost:5000 in your browser.

**What MLflow Tracks:**
- **Parameters**: All hyperparameters (batch size, learning rate, model architecture, etc.)
- **Metrics**: Training loss, validation mAP, learning rate, memory usage
- **Artifacts**: Model checkpoints (automatically saved)
- **Experiments**: Compare multiple training runs side-by-side

**MLflow Features:**
- Compare different runs and hyperparameter combinations
- Filter and search runs by parameters/metrics
- Download model checkpoints from the UI
- Track experiment history automatically

#### Checkpoints
Checkpoints are saved to `models/checkpoints/`:
- `checkpoint_epoch_N.pth`: Full checkpoints (every 10 epochs)
- `checkpoint_epoch_N_lightweight.pth`: Lightweight checkpoints (every epoch)
- `best_model.pth`: Best model based on validation mAP
- `latest_checkpoint.pth`: Most recent checkpoint

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
- Validation every 10 epochs (configurable)

### Experiment Tracking
- **TensorBoard**: Real-time metrics visualization (`logs/`)
- **MLflow**: Comprehensive experiment tracking (`mlruns/`)
  - Automatic parameter and metric logging
  - Model checkpoint artifact storage
  - Experiment comparison and filtering
  - Web UI for visualization

## Troubleshooting

### Out of Memory
- Reduce `batch_size` in `configs/training.yaml`
- Reduce image size in augmentation config
- Gradient accumulation is already implemented (default: 2 steps)
- Reduce `num_workers` and `prefetch_factor` in dataset config
- Memory cleanup runs automatically every N batches

### Slow Training
- Ensure GPU is being used (check CUDA availability)
- Mixed precision training is enabled by default (AMP)
- Model compilation is enabled by default (torch.compile)
- Adjust `num_workers` based on CPU cores (default: 6)
- Check memory usage - if RAM is high, reduce DataLoader workers

### Poor Performance
- Increase training epochs
- Adjust learning rate
- Add more data augmentation
- Check dataset quality and annotations

## Expected Training Time

- **GPU (RTX 3090)**: ~2-3 hours for 50 epochs
- **GPU (RTX 2080)**: ~4-5 hours for 50 epochs
- **CPU**: Not recommended (very slow)

## MLflow Configuration

MLflow is configured in `configs/training.yaml`:

```yaml
logging:
  mlflow: true  # Enable MLflow tracking
  mlflow_tracking_uri: "file:./mlruns"  # Local file storage
  mlflow_experiment_name: "detr_training"  # Experiment name
```

**Tracked Parameters:**
- Training hyperparameters (batch size, learning rate, epochs, etc.)
- Model architecture (backbone, layers, dimensions)
- Dataset information (sample counts)
- Optimization settings (gradient accumulation, mixed precision, etc.)

**Tracked Metrics:**
- `train_loss`: Training loss over steps
- `learning_rate`: Learning rate schedule
- `val_map`: Validation mAP per epoch
- `memory_*`: RAM and GPU memory usage

**Artifacts:**
- Model checkpoints are automatically logged to MLflow
- Access via MLflow UI or programmatically

**Using MLflow Programmatically:**
```python
import mlflow

# Search runs
runs = mlflow.search_runs(experiment_names=["detr_training"])
best_run = runs.loc[runs['metrics.val_map'].idxmax()]

# Load model from MLflow
model = mlflow.pytorch.load_model(f"runs:/{best_run.run_id}/model")
```

## Next Steps

After training:
1. View results in MLflow UI: `./scripts/start_mlflow_ui.sh`
2. Compare different runs and hyperparameter combinations
3. Evaluate model on test set
4. Fine-tune hyperparameters based on MLflow insights
5. Export best model for production use
6. Update inference pipeline to use local model
