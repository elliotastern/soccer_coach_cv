# Adaptive Resource Optimization

## Overview

The adaptive optimizer monitors real-time resource usage (GPU utilization, RAM usage, data loading vs processing times) and provides recommendations for optimal training parameters.

## How It Works

### Monitoring
- **GPU Utilization**: Tracks GPU usage percentage
- **RAM Usage**: Monitors system RAM consumption
- **Timing Metrics**: Tracks data loading time vs GPU processing time
- **Bottleneck Detection**: Identifies if data loading or GPU processing is the bottleneck

### Adjustment Strategies

1. **Low GPU Utilization + Available RAM**
   - Increases `num_workers` (up to max)
   - Increases `prefetch_factor` (up to max)
   - Goal: Keep GPU busy by prefetching more data

2. **Data Loading Bottleneck**
   - If data loading time > GPU processing time
   - Increases `prefetch_factor` to reduce GPU idle time

3. **High RAM Usage**
   - Reduces `num_workers` and `prefetch_factor`
   - Prevents RAM overflow

4. **High GPU Utilization (>95%)**
   - May indicate data-limited scenario
   - Increases prefetch to ensure GPU stays fed

### Limitations

**Important**: PyTorch DataLoaders cannot be recreated during training. The adaptive optimizer:
- ✅ Monitors and logs recommendations
- ✅ Provides real-time metrics
- ✅ Logs optimal settings to MLflow
- ❌ Cannot change DataLoader settings mid-training

### Recommendations

The optimizer logs recommended settings. For the **next training run**, use these optimized values in `configs/training.yaml`:

```yaml
dataset:
  num_workers: <recommended_value>  # From adaptive optimizer
  prefetch_factor: <recommended_value>  # From adaptive optimizer
```

## Configuration

Enable in `configs/training.yaml`:

```yaml
training:
  adaptive_optimization: true  # Enable adaptive monitoring
  target_gpu_utilization: 0.85  # Target GPU utilization (85%)
  max_ram_usage: 0.80  # Maximum RAM usage threshold (80%)
  adaptive_adjustment_interval: 50  # Check every N batches
```

## Output

During training, you'll see:
```
🔧 Adaptive Optimization Adjustment (batch 150):
   num_workers: 9
   prefetch_factor: 4
   GPU util: 72.3%, RAM: 45.2%
   Data load: 0.234s, GPU process: 0.189s
```

At end of each epoch:
```
📊 Adaptive Optimization Stats:
   Adjustments made: 3
   Current workers: 9, prefetch: 4
   Avg GPU util: 78.5%, Avg RAM: 48.3%
```

## Benefits

1. **Real-time Monitoring**: See actual resource usage during training
2. **Bottleneck Identification**: Know if data loading or GPU is the bottleneck
3. **Optimization Recommendations**: Get optimal settings for next run
4. **MLflow Integration**: All metrics logged for analysis

## Best Practices

1. **First Run**: Let adaptive optimizer monitor and recommend
2. **Second Run**: Use recommended settings from first run
3. **Iterate**: Continue optimizing based on recommendations
4. **Monitor**: Check MLflow for adaptive optimization metrics

## Example Workflow

1. Start training with conservative settings (6 workers, prefetch 2)
2. Adaptive optimizer monitors and recommends (e.g., 10 workers, prefetch 4)
3. Check recommendations in logs or MLflow
4. Update `configs/training.yaml` with recommended values
5. Restart training with optimized settings
6. Repeat until optimal balance is found
