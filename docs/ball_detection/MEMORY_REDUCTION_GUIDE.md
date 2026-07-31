# Memory Reduction Guide - 20% Reduction Applied

## Overview
Training configuration has been optimized to reduce system memory requirements by approximately 20% while maintaining training quality.

## Changes Applied

### 1. Resolution Reduction (Biggest Impact)
- **Original:** 1288x1288
- **New:** 1152x1152
- **Memory Impact:** ~20% reduction in activation memory
- **Rationale:** Image area reduced from 1,658,944 to 1,327,104 pixels (80% of original)
- **Note:** 1152 is divisible by 64, maintaining efficiency

### 2. Multi-Scale Training Disabled
- **Original:** `multi_scale: true`
- **New:** `multi_scale: false`
- **Memory Impact:** Significant savings (no multiple scale processing)
- **Rationale:** Reduces memory by avoiding processing at multiple resolutions

### 3. Expanded Scales Disabled
- **Original:** `expanded_scales: true`
- **New:** `expanded_scales: false`
- **Memory Impact:** Additional memory savings
- **Rationale:** Prevents memory overhead from expanded scale processing

### 4. Data Loading Optimizations
- **num_workers:** 2 → 1 (50% reduction)
- **pin_memory:** true → false (saves RAM)
- **prefetch_factor:** default → 1 (less prefetched data)
- **persistent_workers:** false (saves memory)
- **Memory Impact:** Reduces data loading memory footprint

### 5. Gradient Accumulation Adjusted
- **Original:** 16 steps
- **New:** 20 steps
- **Rationale:** Maintains effective batch size (2*20=40 vs 2*16=32) without increasing memory
- **Note:** Gradient accumulation doesn't increase memory, only computation time

## Expected Memory Reduction

### Total Reduction: ~20-25%

Breakdown:
- Resolution reduction: ~20% of activation memory
- Multi-scale disabled: ~5-10% additional savings
- Data loading optimizations: ~3-5% additional savings
- **Total: ~20-25% system memory reduction**

## Training Quality Impact

### Minimal Impact Expected
- **Resolution:** 1152 is still high resolution, sufficient for ball detection
- **Multi-scale:** Disabled, but single-scale training is standard and effective
- **Batch size:** Maintained at 2 (same as original)
- **Effective batch size:** Increased to 40 (from 32) via gradient accumulation

## Usage

### Option 1: Use Low Memory Config
```bash
cd /workspace/soccer_cv_ball
./scripts/resume_training_low_memory.sh
```

### Option 2: Manual Command
```bash
cd /workspace/soccer_cv_ball
python scripts/train_ball.py \
    --config configs/resume_20_epochs_low_memory.yaml \
    --output-dir models
```

## Monitoring Memory Usage

### Check System Memory
```bash
# Monitor system RAM
free -h

# Monitor GPU memory (if using GPU)
nvidia-smi
```

### If Still Running Out of Memory

Further reductions possible:
1. **Reduce resolution further:** 1152 → 1024 (additional ~20% reduction)
2. **Reduce batch size:** 2 → 1 (50% reduction, but slower training)
3. **Disable gradient accumulation:** Reduces effective batch but saves some memory
4. **Enable gradient checkpointing:** Trade compute for memory (if supported)

## Configuration File

The low memory configuration is saved at:
- `configs/resume_20_epochs_low_memory.yaml`

## Comparison

| Setting | Original | Low Memory | Reduction |
|---------|----------|------------|-----------|
| Resolution | 1288 | 1152 | 20% |
| Multi-scale | true | false | - |
| Expanded scales | true | false | - |
| num_workers | 2 | 1 | 50% |
| pin_memory | true | false | - |
| prefetch_factor | default | 1 | - |
| grad_accum_steps | 16 | 20 | - (maintains effective batch) |

## Notes

- The checkpoint will be loaded correctly despite resolution change (RF-DETR handles this)
- Training will resume from epoch 20
- All other training parameters remain the same (learning rate, optimizer, etc.)
- Model architecture unchanged (RF-DETR Base)
