# MLflow Experiment Tracking Guide

This guide explains how to use MLflow for tracking and managing training experiments.

## Overview

MLflow is integrated into the training pipeline to automatically track:
- **Parameters**: Hyperparameters, model architecture, dataset info
- **Metrics**: Training loss, validation mAP, learning rate, memory usage
- **Artifacts**: Model checkpoints

## Quick Start

### 1. Start Training

MLflow tracking is enabled by default. Just start training:

```bash
python scripts/train_detr.py \
    --config configs/training.yaml \
    --train-dir datasets/train \
    --val-dir datasets/val \
    --output-dir models
```

MLflow will automatically:
- Create a new experiment run
- Log all hyperparameters
- Track metrics during training
- Save model checkpoints as artifacts

### 2. View Results

Start the MLflow UI:

```bash
./scripts/start_mlflow_ui.sh
```

Or manually:

```bash
mlflow ui --backend-store-uri file:./mlruns
```

Open http://localhost:5000 in your browser.

## MLflow UI Features

### Experiment View
- See all training runs in the `detr_training` experiment
- Compare runs side-by-side
- Filter runs by parameters or metrics
- Sort by validation mAP or other metrics

### Run Details
- View all logged parameters
- See metric plots over time
- Download model checkpoints
- View training logs

### Comparing Runs
1. Select multiple runs (checkboxes)
2. Click "Compare" to see side-by-side comparison
3. Compare parameters, metrics, and artifacts
4. Identify best hyperparameter combinations

## Tracked Information

### Parameters

**Training Hyperparameters:**
- `batch_size`: Batch size for training
- `learning_rate`: Initial learning rate
- `num_epochs`: Total training epochs
- `weight_decay`: Weight decay for optimizer
- `gradient_clip`: Gradient clipping threshold
- `gradient_accumulation_steps`: Gradient accumulation steps
- `mixed_precision`: Whether AMP is enabled
- `compile_model`: Whether torch.compile is used
- `channels_last`: Memory format optimization

**Model Architecture:**
- `model_architecture`: Model type (detr)
- `backbone`: Backbone network (resnet50)
- `num_classes`: Number of object classes
- `hidden_dim`: Hidden dimension size
- `num_encoder_layers`: Number of encoder layers
- `num_decoder_layers`: Number of decoder layers

**Dataset:**
- `train_samples`: Number of training samples
- `val_samples`: Number of validation samples
- `num_workers`: DataLoader workers
- `prefetch_factor`: DataLoader prefetch factor

### Metrics

**Training Metrics:**
- `train_loss`: Total training loss (logged every N steps)
- `train_loss_ce`: Classification loss component (logged every N steps)
- `train_loss_bbox`: Bounding box regression loss component (logged every N steps)
- `train_loss_giou`: Generalized IoU loss component (logged every N steps)
- `learning_rate`: Current learning rate (logged every N steps)
- `memory_ram_gb`: System RAM usage (logged periodically)
- `memory_gpu_gb`: GPU memory usage (logged periodically)
- `memory_gpu_reserved_gb`: GPU reserved memory (logged periodically)

**Validation Metrics (logged every 10 epochs):**
- `val_map`: Overall validation Mean Average Precision (mAP)
- `val_precision`: Overall validation precision score
- `val_recall`: Overall validation recall score
- `val_f1`: Overall validation F1 score

**Per-Class Validation Metrics (logged every 10 epochs):**
- `val_player_map`: Player class mAP
- `val_player_precision`: Player class precision
- `val_player_recall`: Player class recall
- `val_player_f1`: Player class F1 score
- `val_ball_map`: Ball class mAP
- `val_ball_precision`: Ball class precision
- `val_ball_recall`: Ball class recall
- `val_ball_f1`: Ball class F1 score

These per-class metrics allow you to track performance separately for players and balls, helping identify if one class is learning better than the other.

### Artifacts

- **Checkpoints**: Model checkpoints are saved as artifacts
  - Full checkpoints (every 10 epochs)
  - Best model checkpoint
  - Accessible via MLflow UI or API

- **Models**: Models saved in MLflow's native PyTorch format
  - **Every epoch**: Model saved at `models/epoch_{N}/` for each epoch
  - **Best model**: Also saved at `model/` path for easy access
  - Can be loaded directly with `mlflow.pytorch.load_model()`
  - Includes model metadata (epoch, mAP, is_best flag, config)

## Configuration

Edit `configs/training.yaml` to configure MLflow:

```yaml
logging:
  mlflow: true  # Enable/disable MLflow
  mlflow_tracking_uri: "file:./mlruns"  # Storage location
  mlflow_experiment_name: "detr_training"  # Experiment name
```

### Tracking URI Options

**Local File Storage (Default):**
```yaml
mlflow_tracking_uri: "file:./mlruns"
```

**SQLite Database:**
```yaml
mlflow_tracking_uri: "sqlite:///mlflow.db"
```

**Remote Server:**
```yaml
mlflow_tracking_uri: "http://your-mlflow-server:5000"
```

## Programmatic Access

### Search Runs

```python
import mlflow

# Search all runs in experiment
runs = mlflow.search_runs(experiment_names=["detr_training"])

# Filter by parameters
runs = mlflow.search_runs(
    experiment_names=["detr_training"],
    filter_string="params.batch_size = '24'"
)

# Sort by validation mAP
best_runs = runs.sort_values('metrics.val_map', ascending=False)
```

### Load Model from MLflow

```python
import mlflow.pytorch

# Load best model (saved at standard "model" path)
best_run_id = best_runs.iloc[0]['run_id']
model = mlflow.pytorch.load_model(f"runs:/{best_run_id}/model")

# Or load model from specific epoch
model = mlflow.pytorch.load_model(f"runs:/{run_id}/models/epoch_10")

# Get all available models for a run
import mlflow
run = mlflow.get_run(run_id)
# Check artifacts to see all saved models
```

### Get Run Metrics

```python
import mlflow

# Get specific run
run = mlflow.get_run(run_id)

# Access metrics
val_map = run.data.metrics['val_map']
train_loss = run.data.metrics['train_loss']

# Access parameters
batch_size = run.data.params['batch_size']
learning_rate = run.data.params['learning_rate']
```

## Best Practices

1. **Use Descriptive Experiment Names**: Create separate experiments for different model architectures or datasets
2. **Tag Important Runs**: Use MLflow tags to mark important runs (e.g., "baseline", "best_model")
3. **Compare Before Training**: Check previous runs to avoid repeating experiments
4. **Regular Checkpoints**: Checkpoints are automatically logged - no manual intervention needed
5. **Clean Up Old Runs**: Periodically archive or delete old runs to save space

## Troubleshooting

### MLflow UI Not Starting
```bash
# Check if port 5000 is available
lsof -i :5000

# Use different port
mlflow ui --backend-store-uri file:./mlruns --port 5001
```

### Missing Metrics
- Ensure `mlflow: true` in config
- Check that training completed successfully
- Verify MLflow logs don't show errors

### Large Artifact Storage
- Checkpoints can be large (~160MB each)
- Consider using remote storage for production
- Clean up old checkpoints periodically

## Integration with Other Tools

### TensorBoard
MLflow and TensorBoard work together:
- TensorBoard: Real-time visualization during training
- MLflow: Experiment tracking and comparison

Both are enabled by default and complement each other.

### Export to Production
```python
# Get best model from MLflow
best_run = mlflow.search_runs(
    experiment_names=["detr_training"]
).sort_values('metrics.val_map', ascending=False).iloc[0]

# Export for production
mlflow.pytorch.save_model(
    model,
    "models/production/detr_best",
    registered_model_name="detr_player_ball_detector"
)
```

## Additional Resources

- [MLflow Documentation](https://www.mlflow.org/docs/latest/index.html)
- [MLflow PyTorch Integration](https://www.mlflow.org/docs/latest/python_api/mlflow.pytorch.html)
- [Experiment Tracking Best Practices](https://www.mlflow.org/docs/latest/tracking.html)
