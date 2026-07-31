# Resume Training with 20% Memory Reduction

## Quick Start

```bash
cd /workspace/soccer_cv_ball
./scripts/resume_training_low_memory.sh
```

## Memory Optimizations Applied

### 1. Resolution Reduction (Primary)
- **1288x1288 → 1152x1152**
- **Impact:** ~20% reduction in activation memory
- **Rationale:** Image area reduced from 1.66M to 1.33M pixels (80% of original)

### 2. Multi-Scale Training Disabled
- **Impact:** Significant memory savings
- **Rationale:** Avoids processing at multiple resolutions simultaneously

### 3. Data Loading Optimizations
- **num_workers:** 2 → 1 (50% reduction)
- **pin_memory:** Disabled (saves RAM)
- **prefetch_factor:** Reduced to 1
- **persistent_workers:** Disabled

### 4. Gradient Accumulation
- **Increased:** 16 → 20 steps
- **Rationale:** Maintains effective batch size (40) without increasing memory

## Expected Results

- **Memory Reduction:** ~20-25% system memory
- **Training Quality:** Minimal impact (1152 is still high resolution)
- **Training Speed:** Slightly slower due to reduced num_workers, but manageable

## Configuration

The low memory configuration is at:
- `configs/resume_20_epochs_low_memory.yaml`

## Monitoring

Monitor memory usage:
```bash
# System RAM
free -h

# GPU memory (if using GPU)
nvidia-smi
```

## If Still Running Out of Memory

1. Reduce resolution further: 1152 → 1024
2. Reduce batch size: 2 → 1 (slower but saves memory)
3. Disable gradient accumulation (reduces effective batch)

## Files Created

- `configs/resume_20_epochs_low_memory.yaml` - Low memory config
- `scripts/resume_training_low_memory.sh` - Resume script
- `MEMORY_REDUCTION_GUIDE.md` - Detailed guide
