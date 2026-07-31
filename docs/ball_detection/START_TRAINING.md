# Start Training with Low Memory Configuration

## Quick Start

Training has been configured with **20% memory reduction**. To start training:

```bash
cd /workspace/soccer_cv_ball

# Option 1: Run in foreground (see output)
python scripts/train_ball.py \
    --config configs/resume_20_epochs_low_memory.yaml \
    --output-dir models

# Option 2: Run in background (detached)
nohup python scripts/train_ball.py \
    --config configs/resume_20_epochs_low_memory.yaml \
    --output-dir models > training.log 2>&1 &

# Option 3: Use the resume script
./scripts/resume_training_low_memory.sh
```

## Memory Optimizations Applied

✅ **Resolution:** 1288 → 1152 (20% reduction)  
✅ **Multi-scale:** Disabled  
✅ **Expanded scales:** Disabled  
✅ **num_workers:** 2 → 1  
✅ **pin_memory:** Disabled  
✅ **prefetch_factor:** Reduced to 1  

## Configuration Files

- **Config:** `configs/resume_20_epochs_low_memory.yaml`
- **Checkpoint:** `/workspace/soccer_cv_ball/models/soccer ball/checkpoint_20_soccer_ball.pth`
- **Dataset:** `/workspace/soccer_cv_ball/models/ball_detection_combined_optimized/dataset/`

## Training Details

- **Starting Epoch:** 20 (resuming from checkpoint)
- **Total Epochs:** 40 (will train 20 more epochs)
- **Batch Size:** 2
- **Effective Batch:** 40 (via gradient accumulation: 2 × 20)
- **Learning Rate:** 0.0002
- **Resolution:** 1152x1152

## Monitor Training

```bash
# Check if training is running
ps aux | grep train_ball

# Monitor GPU memory (if using GPU)
watch -n 1 nvidia-smi

# Monitor system memory
watch -n 1 free -h

# View training logs (if running in background)
tail -f training.log
```

## Expected Output

Training should:
1. Load the checkpoint from epoch 19
2. Resume from epoch 20
3. Train for 20 more epochs (20-40)
4. Save checkpoints to `models/checkpoints/`

## Troubleshooting

If you encounter memory issues:
1. Reduce resolution further: 1152 → 1024
2. Reduce batch size: 2 → 1
3. Disable gradient accumulation

If dataset not found:
- Check that COCO dataset exists at the configured path
- Update `coco_train_path` and `coco_val_path` in the config

## Files Created

- `configs/resume_20_epochs_low_memory.yaml` - Low memory config
- `scripts/resume_training_low_memory.sh` - Resume script
- `MEMORY_REDUCTION_GUIDE.md` - Detailed memory guide
- `RESUME_TRAINING_LOW_MEMORY.md` - Quick reference
