# Resume Training from 20-Epoch Checkpoint

## Checkpoint Information

**Checkpoint File:** `/workspace/soccer_cv_ball/models/soccer ball/checkpoint_20_soccer_ball.pth`  
**Size:** 474.57 MB  
**Epoch:** 19 (completed 20 epochs, 0-19)  
**Next Epoch:** 20

## Training Configuration

### Model Architecture
- **Model Type:** RF-DETR Base
- **Encoder:** dinov2_windowed_small
- **Resolution:** 1288x1288
- **Classes:** 2 (ball + background)
- **Class Names:** ['ball']
- **Num Queries:** 300
- **Decoder Layers:** 3
- **Hidden Dim:** 256
- **Self-Attention Heads:** 8
- **Cross-Attention Heads:** 16

### Training Hyperparameters
- **Batch Size:** 2
- **Gradient Accumulation Steps:** 16 (effective batch size: 32)
- **Learning Rate:** 0.0002
- **Encoder Learning Rate:** 0.00015
- **Weight Decay:** 0.0001
- **Gradient Clip:** 0.1
- **Total Epochs:** 20
- **Warmup Epochs:** 0.0
- **LR Scheduler:** step
- **LR Drop:** 100 (not reached)
- **Mixed Precision (AMP):** Enabled

### Loss Configuration
- **Classification Loss Coef:** 1.0
- **Bbox Loss Coef:** 5
- **GIoU Loss Coef:** 2
- **Focal Alpha:** 0.25
- **Auxiliary Loss:** Enabled
- **Set Cost Class:** 2
- **Set Cost Bbox:** 5
- **Set Cost GIoU:** 2

### Optimizer & Scheduler
- **Optimizer State:** ✅ Saved in checkpoint
- **Scheduler State:** ✅ Saved in checkpoint
- **EMA Model:** ✅ Saved (decay: 0.993, tau: 100)

### Dataset Information
- **Original Dataset Path:** `/workspace/soccer_coach_cv/models/ball_detection_open_soccer_ball/dataset`
- **Dataset Format:** Roboflow (YOLO converted to COCO)
- **Original Output Dir:** `/workspace/soccer_coach_cv/models/ball_detection_open_soccer_ball`

### Checkpoint Contents
- ✅ Model state dict (487 layers)
- ✅ Optimizer state dict
- ✅ Learning rate scheduler state
- ✅ EMA model state
- ✅ Training arguments
- ✅ Epoch number (19)

## How to Resume Training

### Option 1: Using the Resume Script

```bash
cd /workspace/soccer_cv_ball
python scripts/resume_from_20_epochs.sh
```

### Option 2: Using train_ball.py with Resume Flag

First, update the dataset path in the config or script to match your current dataset location, then:

```bash
cd /workspace/soccer_cv_ball
python scripts/train_ball.py \
    --config configs/resume_20_epochs.yaml \
    --output-dir models
```

### Option 3: Direct RF-DETR Training

If using RF-DETR directly:

```python
from rfdetr import RFDETRBase

# Initialize model
model = RFDETRBase(class_names=['ball'])

# Load checkpoint
checkpoint_path = "/workspace/soccer_cv_ball/models/soccer ball/checkpoint_20_soccer_ball.pth"
checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)

# Load model weights
if 'model' in checkpoint:
    model_state = checkpoint['model']
    if hasattr(model, 'model') and hasattr(model.model, 'model'):
        current_state = model.model.model.state_dict()
        filtered_state = {}
        for key, value in model_state.items():
            if key in current_state and current_state[key].shape == value.shape:
                filtered_state[key] = value
        model.model.model.load_state_dict(filtered_state, strict=False)

# Continue training with RF-DETR's train() method
# (pass resume=checkpoint_path to resume from epoch 20)
```

## Important Notes

1. **Dataset Path:** The original training used a dataset at `/workspace/soccer_coach_cv/models/ball_detection_open_soccer_ball/dataset`. You may need to:
   - Update the dataset path in the config/script to match your current dataset location
   - Or ensure the dataset exists at the original path

2. **Epoch Continuation:** The checkpoint is at epoch 19, so resuming will start from epoch 20. If you want to train for more epochs, update the `epochs` parameter.

3. **Output Directory:** The original training saved to `/workspace/soccer_coach_cv/models/ball_detection_open_soccer_ball`. You may want to change this to save in the current workspace.

4. **Model Compatibility:** The checkpoint uses RF-DETR format with the model structure: `model.model.model` (RFDETRBase -> Model -> LWDETR).

## Files Created

1. **`training_info_20_epochs.json`** - Complete training information extracted from checkpoint
2. **`configs/resume_20_epochs.yaml`** - YAML config for resuming training
3. **`scripts/resume_from_20_epochs.sh`** - Python script to resume training
4. **`scripts/verify_checkpoint.py`** - Script to verify checkpoint validity

## Training Progress

- **Completed:** 20 epochs (0-19)
- **Checkpoint saved:** Epoch 19
- **Ready to resume:** Yes ✅

## Next Steps

1. Verify dataset path exists and is accessible
2. Update paths in config/script if needed
3. Run resume script to continue training from epoch 20
4. Monitor training logs and metrics
