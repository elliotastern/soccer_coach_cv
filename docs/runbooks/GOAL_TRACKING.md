# Performance Goals and Tracking

This document describes the performance goals for the soccer object detection model and how they are tracked in MLflow.

## Performance Goals

### Player Detection Goals
- **Player Detection Recall**: > 95% (IoU ≥ 0.5) - Must detect players even in clusters
- **Player Detection Precision**: > 80% (IoU ≥ 0.5) - Balance with recall to reduce false positives
- **Player Detection mAP@0.5**: > 85% - Mean Average Precision at IoU 0.5
- **Player Detection mAP@0.75**: > 70% - Stricter localization quality

### Ball Detection Goals
- **Ball Detection Recall**: ~80% (IoU ≥ 0.5) - Allowing for motion blur/occlusion (to be fixed by interpolation later)
- **Ball Detection Precision**: > 70% (IoU ≥ 0.5) - Reduce false positives
- **Ball Detection mAP@0.5**: > 70% - Mean Average Precision at IoU 0.5
- **Ball Detection Count**: > 0 predictions per validation image with balls - Diagnostic metric

## MLflow Tracking

### Goal Parameters
Goals are logged as parameters at the start of each training run:
- `goal_player_recall_05`: 0.95
- `goal_player_precision_05`: 0.80
- `goal_player_map_05`: 0.85
- `goal_player_map_75`: 0.70
- `goal_ball_recall_05`: 0.80
- `goal_ball_precision_05`: 0.70
- `goal_ball_map_05`: 0.70
- `goal_ball_avg_predictions_per_image`: 1.0

### Goal Achievement Metrics
Logged every validation epoch (every 10 epochs):
- `goal_player_recall_05_achieved`: 1.0 if achieved, 0.0 if not
- `goal_player_precision_05_achieved`: 1.0 if achieved, 0.0 if not
- `goal_player_map_05_achieved`: 1.0 if achieved, 0.0 if not
- `goal_player_map_75_achieved`: 1.0 if achieved, 0.0 if not
- `goal_ball_recall_05_achieved`: 1.0 if achieved, 0.0 if not
- `goal_ball_precision_05_achieved`: 1.0 if achieved, 0.0 if not
- `goal_ball_map_05_achieved`: 1.0 if achieved, 0.0 if not
- `goal_ball_avg_predictions_achieved`: 1.0 if achieved, 0.0 if not

### Goal Progress Metrics
Logged every validation epoch, showing percentage progress toward each goal:
- `goal_player_recall_05_progress`: 0-100% (capped at 100%)
- `goal_player_precision_05_progress`: 0-100%
- `goal_player_map_05_progress`: 0-100%
- `goal_player_map_75_progress`: 0-100%
- `goal_ball_recall_05_progress`: 0-100%
- `goal_ball_precision_05_progress`: 0-100%
- `goal_ball_map_05_progress`: 0-100%
- `goal_ball_avg_predictions_progress`: 0-100%

## Using Goal Tracking in MLflow

### Viewing Goals
1. Open MLflow UI: `mlflow ui --backend-store-uri file:./mlruns`
2. Select a run
3. Go to "Parameters" tab to see goal values
4. Go to "Metrics" tab to see goal achievement and progress

### Filtering by Goal Achievement
You can filter runs in MLflow UI by goal achievement:
- Search for runs where `goal_player_recall_05_achieved = 1.0`
- Compare progress metrics across runs
- Identify which hyperparameters lead to goal achievement

### Monitoring Progress
- Watch `goal_*_progress` metrics over time to see improvement
- Goal achievement metrics (0.0 or 1.0) show when goals are met
- Progress metrics show how close you are to each goal

## Metrics Calculated

### mAP Calculation
- **mAP@0.5**: Mean Average Precision at IoU threshold 0.5 (standard)
- **mAP@0.75**: Mean Average Precision at IoU threshold 0.75 (stricter)

### Ball Count Metric
- **ball_avg_predictions_per_image**: Average number of ball predictions per image that contains balls
- Helps diagnose if the model is making any ball predictions at all
- Goal: > 1.0 (at least one prediction per image with balls)

## Implementation Details

The evaluator now:
1. Calculates metrics separately at IoU 0.5 and 0.75
2. Tracks ball predictions per image with balls
3. Returns all metrics in a single dictionary

The trainer:
1. Logs goals as parameters at run start
2. Logs goal achievement status (0/1) after each validation
3. Logs goal progress (percentage) after each validation
