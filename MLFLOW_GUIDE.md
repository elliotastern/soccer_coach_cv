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

**Performance Goals:**
- `goal_player_recall_05`: Target player recall at IoU 0.5 (0.95)
- `goal_player_precision_05`: Target player precision at IoU 0.5 (0.80)
- `goal_player_map_05`: Target player mAP at IoU 0.5 (0.85)
- `goal_player_map_75`: Target player mAP at IoU 0.75 (0.70)
- `goal_ball_recall_05`: Target ball recall at IoU 0.5 (0.80)
- `goal_ball_precision_05`: Target ball precision at IoU 0.5 (0.70)
- `goal_ball_map_05`: Target ball mAP at IoU 0.5 (0.70)
- `goal_ball_avg_predictions_per_image`: Target average ball predictions per image (1.0)

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

**Player Metrics (IoU 0.5):**
- `val_player_map_05`: Player class mAP at IoU 0.5
- `val_player_precision_05`: Player class precision at IoU 0.5
- `val_player_recall_05`: Player class recall at IoU 0.5
- `val_player_f1`: Player class F1 score

**Player Metrics (IoU 0.75):**
- `val_player_map_75`: Player class mAP at IoU 0.75 (stricter localization)

**Ball Metrics (IoU 0.5):**
- `val_ball_map_05`: Ball class mAP at IoU 0.5
- `val_ball_precision_05`: Ball class precision at IoU 0.5
- `val_ball_recall_05`: Ball class recall at IoU 0.5
- `val_ball_f1`: Ball class F1 score

**Ball Metrics (IoU 0.75):**
- `val_ball_map_75`: Ball class mAP at IoU 0.75 (stricter localization)

**Ball Detection Count:**
- `val_ball_avg_predictions_per_image`: Average number of ball predictions per image that contains balls
- `val_images_with_balls`: Number of validation images containing balls

**Goal Tracking Metrics:**
- `goal_player_recall_05_achieved`: 1.0 if player recall ≥ 95%, else 0.0
- `goal_player_precision_05_achieved`: 1.0 if player precision ≥ 80%, else 0.0
- `goal_player_map_05_achieved`: 1.0 if player mAP@0.5 ≥ 85%, else 0.0
- `goal_player_map_75_achieved`: 1.0 if player mAP@0.75 ≥ 70%, else 0.0
- `goal_ball_recall_05_achieved`: 1.0 if ball recall ≥ 80%, else 0.0
- `goal_ball_precision_05_achieved`: 1.0 if ball precision ≥ 70%, else 0.0
- `goal_ball_map_05_achieved`: 1.0 if ball mAP@0.5 ≥ 70%, else 0.0
- `goal_ball_avg_predictions_achieved`: 1.0 if avg ball predictions ≥ 1.0 per image, else 0.0

**Goal Progress Metrics:**
- `goal_player_recall_05_progress`: Percentage progress toward 95% recall goal
- `goal_player_precision_05_progress`: Percentage progress toward 80% precision goal
- `goal_player_map_05_progress`: Percentage progress toward 85% mAP@0.5 goal
- `goal_player_map_75_progress`: Percentage progress toward 70% mAP@0.75 goal
- `goal_ball_recall_05_progress`: Percentage progress toward 80% recall goal
- `goal_ball_precision_05_progress`: Percentage progress toward 70% precision goal
- `goal_ball_map_05_progress`: Percentage progress toward 70% mAP@0.5 goal
- `goal_ball_avg_predictions_progress`: Percentage progress toward 1.0 predictions per image goal

These metrics allow you to track performance separately for players and balls at different IoU thresholds, and monitor progress toward your performance goals.

## Understanding Training Metrics

This section explains what each metric measures.

### Learning Rate (`learning_rate`)

**What it is:** The current learning rate used by the optimizer during training. Controls how large steps the model takes when updating weights.

### Total Training Loss (`train_loss`)

**What it is:** The overall training loss, which is the sum of all loss components (classification + bounding box + GIoU). Measures how well the model is performing on training data.

### Classification Loss (`train_loss_ce`)

**What it is:** The cross-entropy loss for object classification. Measures how accurately the model predicts whether an object is a player, ball, or background.

### Bounding Box Regression Loss (`train_loss_bbox`)

**What it is:** The L1 loss for bounding box coordinates. Measures how accurately the model predicts the x, y, width, and height of bounding boxes around objects.

### Generalized IoU Loss (`train_loss_giou`)

**What it is:** The Generalized Intersection over Union (GIoU) loss. Measures how well predicted bounding boxes overlap with ground-truth boxes.

### Validation Metrics

**Mean Average Precision (`val_map`):**
- **What it is:** Overall detection accuracy combining both classification and localization performance across all classes.

**Per-Class Metrics (`val_player_map`, `val_ball_map`, etc.):**
- **What it is:** Detection accuracy for each class separately (players and balls). Helps identify if one class is learning better than another.

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
