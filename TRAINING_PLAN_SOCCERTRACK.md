# Comprehensive Training Plan: SoccerTrack_sub Dataset
## Based on Architectural Optimization Strategic Report

**Dataset**: `/workspace/soccer_coach_cv/data/raw/SoccerTrack_sub`  
**Target**: Train RF-DETR model optimized for small object detection (ball <15 pixels)  
**Hardware**: NVIDIA RTX 5090 (or equivalent high-end GPU)

---

## Phase 1: Data Preparation & Conversion

### 1.1 Dataset Analysis
**Status**: SoccerTrack_sub contains:
- 5 video files (panoramic footage)
- Event labels (PASS, DRIVE, etc.) in JSON format
- **Missing**: Bounding box annotations for players and ball

### 1.2 Data Preparation Strategy

#### Option A: Pseudo-Labeling (Recommended for Speed)
1. **Extract frames** from videos at strategic intervals
   - Extract every Nth frame (e.g., every 30 frames = ~1.2s at 25fps)
   - Focus on frames with events (use JSON position timestamps)
   - Target: ~500-1000 frames per video = 2500-5000 total frames

2. **Generate initial annotations** using pre-trained RF-DETR
   - Use existing model or Roboflow RF-DETR base model
   - Run inference with SAHI (Slicing Aided Hyper Inference) enabled
   - Low confidence threshold (0.1-0.2) to capture all detections
   - Export detections to COCO format

3. **Active Learning Selection**
   - Identify low-confidence frames (0.1-0.4 confidence range)
   - Prioritize frames with ball detections
   - Manual review/refinement of ~200-500 "hard" frames
   - This concentrates human effort where most needed

#### Option B: Manual Annotation (Higher Quality, More Time)
1. Extract frames at event timestamps
2. Annotate in CVAT or similar tool
3. Convert to COCO format

### 1.3 COCO Format Conversion

**Required Structure**:
```
datasets/
├── train/
│   ├── images/
│   │   ├── frame_000001.jpg
│   │   └── ...
│   └── annotations/
│       └── annotations.json
└── val/
    ├── images/
    │   ├── frame_000001.jpg
    │   └── ...
    └── annotations/
        └── annotations.json
```

**COCO JSON Format**:
- Categories: `player` (id=1), `ball` (id=2)
- Bounding boxes: `[x, y, width, height]` (top-left origin)
- Train/Val split: 80/20 or 70/30

**Implementation Script**: Create `scripts/prepare_soccertrack_dataset.py`

---

## Phase 2: Training Configuration Optimization

### 2.1 Model Architecture (configs/training.yaml)

**Current Settings** (to verify/update):
```yaml
model:
  architecture: "rfdetr"  # Use RF-DETR (not vanilla DETR)
  rfdetr_size: "base"    # Optimal for RTX 5090
  num_classes: 2         # player, ball
  pretrained: true       # Essential: use pre-trained weights
```

### 2.2 Hyperparameters Optimization

**Key Changes Based on Strategic Report**:
```yaml
training:
  batch_size: 32  # Increase for RTX 5090 (was 24 for A40)
  num_epochs: 50  # Start conservative, can extend
  learning_rate: 0.0001  # 1e-4 (standard)
  weight_decay: 0.0001  # 1e-4
  warmup_epochs: 5
  gradient_clip: 0.1
  gradient_accumulation_steps: 2
  mixed_precision: true  # FP16/FP32 mixed (CRITICAL for small objects)
  compile_model: false  # Keep disabled (variable input sizes)
  channels_last: true   # Memory optimization
  cudnn_benchmark: true  # Speed optimization
  tf32: true           # Enable TF32 on Ampere+ GPUs
```

### 2.3 Advanced Augmentation Pipeline

**Critical Augmentations for Sim2Real Gap** (from strategic report):

```yaml
augmentation:
  train:
    horizontal_flip: 0.5
    color_jitter:
      brightness: 0.2
      contrast: 0.2
      saturation: 0.2
      hue: 0.1
    resize_range: [800, 1333]  # DETR standard
    
    # CRITICAL: Motion Blur for fast-moving ball
    motion_blur:
      enabled: true
      prob: 0.4  # Increased from 0.3
      max_kernel_size: 20  # Increased for more realistic blur
    
    # CRITICAL: Gaussian Blur for sensor noise
    gaussian_blur:
      enabled: true
      prob: 0.3
      kernel_size_range: [3, 7]
      sigma_range: [0.5, 2.0]
    
    # CRITICAL: ISO Noise Injection
    iso_noise:
      enabled: true
      prob: 0.3
      noise_level: [5, 25]  # Simulate camera noise
    
    # CRITICAL: JPEG Compression Artifacts
    jpeg_compression:
      enabled: true
      prob: 0.2
      quality_range: [60, 95]  # Simulate broadcast compression
    
    # MixUp for occlusion handling
    mixup:
      enabled: true
      prob: 0.3
      alpha: 0.2
    
    # Mosaic for multi-scale learning
    mosaic:
      enabled: true
      prob: 0.5
      min_scale: 0.4
      max_scale: 1.0
    
    # Copy-Paste for ball class balancing
    copy_paste:
      enabled: true
      prob: 0.5
      max_pastes: 3
    
    # CLAHE for contrast enhancement
    clahe:
      enabled: true
      clip_limit: 2.0
      tile_grid_size: [8, 8]
```

**Implementation**: Update `src/training/augmentation.py` to add:
- `GaussianBlurAugmentation`
- `ISONoiseAugmentation`
- `JPEGCompressionAugmentation`
- `MixUpAugmentation`
- `MosaicAugmentation`

### 2.4 Class Imbalance Handling

**Focal Loss Configuration** (already enabled):
```yaml
focal_loss:
  enabled: true
  alpha: 0.5  # INCREASE from 0.25 for ball class
  gamma: 2.0  # Focus on hard examples
```

**Weighted Random Sampling** (NEW - to implement):
- Oversample images containing ball
- Implement `WeightedRandomSampler` in data loader
- Weight formula: `weight = 1.0 if no ball, 3.0 if ball present`

---

## Phase 3: Progressive Training Protocol

### 3.1 Phase 1: Head Training (Epochs 1-10)

**Strategy**: Freeze backbone, train only decoder + heads

**Implementation**:
```python
# Freeze backbone
for param in model.backbone.parameters():
    param.requires_grad = False

# Train only transformer decoder and detection heads
learning_rate = 1e-4
num_epochs = 10
```

**Rationale**: 
- Prevents catastrophic forgetting of pre-trained features
- Fast convergence of detection heads
- Stabilizes training

### 3.2 Phase 2: Full Fine-Tuning (Epochs 11-50)

**Strategy**: Unfreeze backbone, reduce learning rate

**Implementation**:
```python
# Unfreeze backbone
for param in model.backbone.parameters():
    param.requires_grad = True

# Reduce learning rate by 10x
learning_rate = 1e-5  # 10% of original
num_epochs = 40
```

**Rationale**:
- Allows backbone to adapt to soccer-specific textures
- Lower LR prevents destroying pre-trained features
- Fine-tunes for grass, net patterns, etc.

### 3.3 Learning Rate Schedule

**Cosine Annealing with Warmup**:
```yaml
lr_schedule:
  type: "cosine"
  warmup_epochs: 5
  min_lr: 1e-6
```

**Phase-specific adjustments**:
- Phase 1: Standard cosine schedule
- Phase 2: Restart cosine schedule with lower max LR

---

## Phase 4: Training Execution

### 4.1 Pre-Training Checklist

- [ ] Dataset converted to COCO format
- [ ] Train/Val split created (80/20)
- [ ] Augmentation pipeline implemented
- [ ] Config files updated
- [ ] GPU memory verified (RTX 5090 ready)
- [ ] MLflow tracking configured
- [ ] Checkpoint directory created

### 4.2 Training Command

**Phase 1 (Head Training)**:
```bash
python scripts/train_detr.py \
    --config configs/training_soccertrack_phase1.yaml \
    --train-dir datasets/train \
    --val-dir datasets/val \
    --output-dir models/soccertrack_training
```

**Phase 2 (Full Fine-Tuning)**:
```bash
python scripts/train_detr.py \
    --config configs/training_soccertrack_phase2.yaml \
    --train-dir datasets/train \
    --val-dir datasets/val \
    --output-dir models/soccertrack_training \
    --resume models/soccertrack_training/checkpoints/checkpoint_epoch_10_lightweight.pth
```

### 4.3 Monitoring

**MLflow UI**:
```bash
./scripts/start_mlflow_ui.sh
# Or: mlflow ui --backend-store-uri file:./mlruns
```

**Key Metrics to Track**:
- `train_loss`: Should decrease steadily
- `val_map`: Mean Average Precision (target: >0.5 for ball)
- `val_map_ball`: Ball-specific mAP (target: >0.4)
- `val_map_player`: Player-specific mAP (target: >0.7)
- `learning_rate`: Verify schedule
- `memory_usage`: GPU/RAM utilization

**TensorBoard**:
```bash
tensorboard --logdir logs
```

### 4.4 Checkpoint Strategy

**Lightweight Checkpoints** (every epoch):
- Save model state dict only
- Keep last 20 checkpoints
- Path: `models/checkpoints/checkpoint_epoch_N_lightweight.pth`

**Full Checkpoints** (every 10 epochs):
- Save full training state (optimizer, scheduler, etc.)
- For resuming training
- Path: `models/checkpoints/checkpoint_epoch_N.pth`

**Best Model**:
- Tracked by validation mAP
- Saved automatically when new best found
- Path: `models/checkpoints/best_model.pth`

---

## Phase 5: Post-Training Optimization

### 5.1 Model Compilation (Inference Only)

**Note**: `torch.compile` disabled during training (variable input sizes)
**Enable for inference**:
```python
# After training, compile for inference
model = torch.compile(model, mode='reduce-overhead')
```

### 5.2 Precision Optimization

**Training**: Mixed Precision (FP16/FP32) - already enabled
**Inference**: FP16 (half precision)
```python
model.half()  # Convert to FP16
```

**Future**: INT8 via QAT (Quantization-Aware Training) if needed
- **CRITICAL**: Use QAT, NOT PTQ for small objects
- PTQ will destroy small object detection capability

### 5.3 SAHI Integration (Inference)

**For production inference**, wrap model with SAHI:
```python
from sahi import AutoDetectionModel
from sahi.predict import get_sliced_prediction

detection_model = AutoDetectionModel.from_pretrained(
    model_type='rfdetr',
    model_path='models/checkpoints/best_model.pth',
    confidence_threshold=0.1,
    device='cuda:0'
)

# Slice size optimized for 4K footage
result = get_sliced_prediction(
    image,
    detection_model,
    slice_height=1280,
    slice_width=1280,
    overlap_height_ratio=0.2,
    overlap_width_ratio=0.2
)
```

---

## Phase 6: Validation & Testing

### 6.1 Validation Metrics

**Primary Metrics**:
- **mAP@0.5**: Mean Average Precision at IoU=0.5
- **mAP@0.75**: Mean Average Precision at IoU=0.75
- **mAP_ball**: Ball-specific mAP (most critical)
- **mAP_player**: Player-specific mAP

**Target Performance**:
- Ball mAP@0.5: >0.40 (baseline was 0.25)
- Ball mAP@0.75: >0.30
- Player mAP@0.5: >0.70
- Overall mAP@0.5: >0.60

### 6.2 Test on Real Footage

**Test Videos**:
- Use held-out SoccerTrack_sub videos
- Run inference with SAHI enabled
- Evaluate ball recall (should be >80% vs baseline 25%)

**Visualization**:
```bash
python scripts/visualize_predictions.py \
    --model models/checkpoints/best_model.pth \
    --video data/raw/SoccerTrack_sub/videos/117093_panorama_1st_half-017.mp4 \
    --output visualizations/
```

### 6.3 Error Analysis

**Common Failure Modes to Check**:
1. **Ghost Balls**: White socks/lines detected as ball
   - Solution: Parabolic consistency checks (post-processing)
2. **Missed Fast Balls**: Motion blur causes misses
   - Solution: Verify motion blur augmentation worked
3. **Occluded Players**: Players behind others
   - Solution: Verify MixUp/Mosaic augmentation

---

## Implementation Checklist

### Data Preparation
- [ ] Create `scripts/prepare_soccertrack_dataset.py`
  - [ ] Extract frames from videos
  - [ ] Generate pseudo-labels with pre-trained model + SAHI
  - [ ] Convert to COCO format
  - [ ] Create train/val split
- [ ] Verify dataset structure
- [ ] Validate COCO annotations

### Augmentation Implementation
- [ ] Add `GaussianBlurAugmentation` to `src/training/augmentation.py`
- [ ] Add `ISONoiseAugmentation`
- [ ] Add `JPEGCompressionAugmentation`
- [ ] Add `MixUpAugmentation`
- [ ] Add `MosaicAugmentation`
- [ ] Update `get_train_transforms()` to include all augmentations
- [ ] Test augmentation pipeline

### Training Configuration
- [ ] Create `configs/training_soccertrack_phase1.yaml`
  - [ ] Freeze backbone settings
  - [ ] Phase 1 hyperparameters
  - [ ] Augmentation config
- [ ] Create `configs/training_soccertrack_phase2.yaml`
  - [ ] Unfreeze backbone settings
  - [ ] Phase 2 hyperparameters (lower LR)
  - [ ] Resume from Phase 1 checkpoint
- [ ] Update Focal Loss alpha to 0.5

### Progressive Training Implementation
- [ ] Modify `scripts/train_detr.py` or create wrapper
  - [ ] Add backbone freeze/unfreeze logic
  - [ ] Add phase-specific learning rate handling
  - [ ] Add checkpoint-based phase transition
- [ ] Test Phase 1 training (head-only)
- [ ] Test Phase 2 training (full fine-tuning)

### Class Imbalance Handling
- [ ] Implement `WeightedRandomSampler` in data loader
- [ ] Calculate weights based on ball presence
- [ ] Integrate with `CocoDataset` class

### Monitoring & Validation
- [ ] Verify MLflow tracking
- [ ] Set up TensorBoard logging
- [ ] Create validation script with detailed metrics
- [ ] Create visualization script for predictions

### Post-Training
- [ ] Export best model
- [ ] Test inference with SAHI
- [ ] Benchmark inference speed
- [ ] Create inference pipeline with optimizations

---

## Expected Timeline

**Data Preparation**: 2-4 hours
- Frame extraction: 1 hour
- Pseudo-labeling: 1-2 hours
- Manual review (200 frames): 1 hour
- COCO conversion: 30 minutes

**Augmentation Implementation**: 2-3 hours
- Code implementation: 1-2 hours
- Testing: 1 hour

**Training Configuration**: 1 hour
- Config files: 30 minutes
- Testing: 30 minutes

**Phase 1 Training**: 2-4 hours (10 epochs)
**Phase 2 Training**: 8-16 hours (40 epochs)

**Total Training Time**: 10-20 hours (depending on dataset size)

**Validation & Testing**: 2-3 hours

**Total Project Time**: ~20-30 hours

---

## Success Criteria

1. **Ball Detection Recall**: >80% (vs baseline 25%)
2. **Ball mAP@0.5**: >0.40 (vs baseline ~0.15)
3. **Player mAP@0.5**: >0.70
4. **Inference Speed**: Real-time or near-real-time with SAHI on RTX 5090
5. **No "Invisible Ball" failures**: Ball detected in >80% of frames where visible

---

## Next Steps After Training

1. **ByteTrack Integration**: Implement hybrid tracking (ByteTrack for ball, Deep-EIoU for players)
2. **Parabolic Consistency Checks**: Post-processing to filter ghost balls
3. **Team Identification**: Color clustering for team assignment
4. **Event Detection**: Heuristic logic layer for pass/shot/dribble detection
5. **Production Deployment**: Optimize inference pipeline with torch.compile + SAHI

---

## References

- Strategic Report: "Architectural Optimization of Real-Time Computer Vision Pipelines for Sports Analytics"
- RF-DETR Documentation: Roboflow Model Zoo
- SAHI Documentation: https://github.com/obss/sahi
- DETR Paper: "End-to-End Object Detection with Transformers"
- ByteTrack Paper: "ByteTrack: Multi-Object Tracking by Associating Every Detection Box"
