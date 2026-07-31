# Resume Training Configuration Files

This directory contains alternative training configurations based on the comprehensive evaluation.

## Base Configuration

**`resume_20_epochs_low_memory.yaml`** - Current configuration
- Resolution: 1120 (reduced for memory)
- Multi-scale: Disabled
- Learning rate: 0.0002 (fixed)
- Batch size: 2
- **Status**: Working well, metrics improving

## Alternative Configurations

### 1. Multi-scale Training (`resume_with_multiscale.yaml`)

**Purpose**: Improve small object (ball) detection by enabling multi-scale training

**Changes**:
- `multi_scale: true` (enabled)
- `epochs: 50` (extended)
- `start_epoch: 40` (resume from epoch 40)

**Benefits**:
- Helps model learn at multiple scales
- Particularly beneficial for small objects
- May improve small objects mAP from 0.598 to >0.65

**Risks**:
- May increase GPU memory usage
- Monitor memory and reduce batch_size if needed

**When to Use**:
- Small objects mAP < 0.65
- Overall mAP > 0.68 (good overall performance)
- GPU memory allows for multi-scale training

### 2. Higher Resolution (`resume_with_higher_resolution.yaml`)

**Purpose**: Improve small object detection by increasing image resolution

**Changes**:
- `resolution: 1288` (increased from 1120)
- `epochs: 50` (extended)
- `start_epoch: 40` (resume from epoch 40)

**Benefits**:
- Higher resolution preserves small object details
- Better for detecting tiny balls
- May improve small objects mAP from 0.598 to >0.65

**Risks**:
- Significantly increases GPU memory usage (~15-20% more)
- May cause OOM errors
- Slower training (more pixels to process)
- May need to reduce batch_size to 1

**When to Use**:
- Small objects mAP < 0.65
- GPU has sufficient memory (monitor closely)
- Multi-scale not feasible due to memory

## Learning Rate Schedule

**Note**: RF-DETR may have built-in learning rate scheduling. The current config doesn't explicitly specify LR schedule, so RF-DETR may be using defaults.

If metrics plateau, consider:
1. Checking RF-DETR documentation for LR schedule parameters
2. Manually reducing learning rate in config (e.g., 0.0002 → 0.0001)
3. Using step decay: reduce LR by 0.5x at epoch 40, 0.1x at epoch 50

## Usage

### Continue with Current Settings (Recommended)
```bash
python scripts/train_ball.py \
    --config configs/resume_20_epochs_low_memory.yaml \
    --output-dir models \
    --resume models/checkpoint.pth
```

### Try Multi-scale Training
```bash
# First, update start_epoch to 40 in the config after epoch 40 completes
python scripts/train_ball.py \
    --config configs/resume_with_multiscale.yaml \
    --output-dir models \
    --resume models/checkpoint.pth
```

### Try Higher Resolution
```bash
# First, update start_epoch to 40 in the config after epoch 40 completes
# Monitor GPU memory closely - may need to reduce batch_size to 1
python scripts/train_ball.py \
    --config configs/resume_with_higher_resolution.yaml \
    --output-dir models \
    --resume models/checkpoint.pth
```

## Decision Guide

**Use Base Config** if:
- Metrics still improving (>0.001 per epoch)
- No memory constraints
- Small objects improving steadily

**Use Multi-scale Config** if:
- Small objects mAP < 0.65
- Overall mAP > 0.68
- Memory allows for multi-scale

**Use Higher Resolution Config** if:
- Small objects mAP < 0.65
- Multi-scale not feasible
- GPU has sufficient memory

**Don't Change Config** if:
- Metrics plateauing for <3 epochs
- Overfitting detected
- Target metrics achieved

## Evaluation

After each training phase, run:
```bash
python scripts/comprehensive_training_evaluation.py <config_file>
```

This will analyze progress and provide updated recommendations.
