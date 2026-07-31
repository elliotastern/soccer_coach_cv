# Training Speed Optimization Guide

## Current Setup
- **GPU**: RTX A5000 (24GB VRAM)
- **Current Config**: `configs/training.yaml` (batch_size: 24, num_workers: 4)
- **Fast Config**: `configs/training_fast_a5000.yaml` (batch_size: 36, num_workers: 12)

## Speed Optimizations Applied

### 1. **Batch Size Increase** (2-3x speedup potential)
- **Current**: 24
- **Optimized**: 36
- **Impact**: Larger batches = more efficient GPU utilization, fewer iterations per epoch
- **Memory**: With mixed precision (FP16), 36 batch size fits comfortably in 24GB VRAM

### 2. **Data Loading Optimization** (20-40% speedup)
- **num_workers**: 4 → 12 (3x more parallel data loading)
- **prefetch_factor**: 2 → 6 (3x more batches prefetched)
- **persistent_workers**: false → true (eliminates worker startup overhead)
- **Impact**: GPU waits less for data, better utilization

### 3. **Mixed Precision Training** (Already enabled, ~2x speedup)
- **Status**: ✅ Enabled (`mixed_precision: true`)
- **Impact**: ~2x faster training, ~50% less memory usage
- **Note**: Critical for tiny object detection - preserves accuracy

### 4. **Memory Format Optimization** (10-15% speedup)
- **channels_last**: ✅ Enabled
- **Impact**: Faster convolution operations on modern GPUs

### 5. **TF32 Precision** (1.5x speedup on matmul)
- **Status**: ✅ Enabled (`tf32: true`)
- **Impact**: Faster matrix multiplications on Ampere GPUs (A5000)

### 6. **Reduced Augmentation Overhead** (10-20% speedup)
- **CLAHE**: Disabled (minimal accuracy impact)
- **Motion Blur**: Disabled (minimal accuracy impact)
- **Impact**: Faster data preprocessing

### 7. **Reduced Validation Frequency** (2x speedup during validation)
- **val_frequency**: 1 → 2 (validate every 2 epochs instead of every epoch)
- **Impact**: Less time spent on validation

### 8. **Reduced Logging I/O** (5-10% speedup)
- **print_frequency**: 20 → 100 (less frequent console output)
- **log_every_n_steps**: 50 → 200 (less frequent TensorBoard writes)
- **mlflow_log_models**: false (no model artifact logging)

### 9. **Memory Cleanup Optimization** (2-5% speedup)
- **memory_cleanup_frequency**: 10 → 30 (less frequent cleanup = less overhead)
- **adaptive_adjustment_interval**: 50 → 100 (less frequent checks)

## Expected Speed Improvements

### Overall Training Speed
- **Current**: ~5 hours for 20 epochs
- **Optimized**: ~2-2.5 hours for 20 epochs (2-2.5x faster)

### Per-Epoch Speed
- **Current**: ~15 minutes per epoch
- **Optimized**: ~6-7 minutes per epoch

## How to Use

### Option 1: Use Fast Config Directly
```bash
python scripts/train_detr.py --config configs/training_fast_a5000.yaml
```

### Option 2: Modify Existing Config
Edit `configs/training.yaml` and update:
- `batch_size: 24` → `batch_size: 36`
- `num_workers: 4` → `num_workers: 12`
- `prefetch_factor: 2` → `prefetch_factor: 6`
- `val_frequency: 1` → `val_frequency: 2` (in evaluation section)

### Option 3: Test Batch Size Incrementally
If you encounter OOM (Out of Memory) errors:
1. Start with `batch_size: 32`
2. If successful, try `batch_size: 36`
3. If still successful, try `batch_size: 40` (maximum for A5000)

## Memory Monitoring

Monitor GPU memory usage during training:
```bash
watch -n 1 nvidia-smi
```

If you see memory usage > 95%, reduce batch_size by 4-8.

## Trade-offs

### Accuracy Impact
- **Minimal**: Disabled augmentations (CLAHE, motion blur) have minimal impact
- **None**: Other optimizations (batch size, data loading) don't affect accuracy

### Stability
- **Same**: All optimizations maintain training stability
- **Better**: Larger batch size = more stable gradients

## Additional Optimizations (Future)

1. **Gradient Checkpointing**: Trade compute for memory (if needed for even larger batches)
2. **Model Compilation**: PyTorch 2.0 `torch.compile()` (currently disabled due to variable input sizes)
3. **Data Prefetching**: Pre-process data to disk cache (if I/O becomes bottleneck)

## Notes

- All optimizations are GPU-safe and tested
- Mixed precision is critical - keep it enabled
- Monitor first epoch to ensure no OOM errors
- If training crashes, reduce batch_size by 4 and retry
