# Small Object Detection Optimization Strategy - REVISED
## Physics-Aware Approach for High-Velocity Ball Detection

**Critical Insight**: The 0% SAHI recall indicates a **Sim2Real domain gap**, not just scale/resolution. Balls at 100km/h appear as **streaks**, not spheres. The current strategy optimizes for the wrong problem.

**Current Status**: Small Objects mAP = 0.598 (59.8%) - Target: 0.65-0.70  
**Root Cause**: Model trained on clean synthetic data, inference on noisy/blurred real video  
**Hardware**: RTX 5090 (32GB VRAM) - **underutilized**, can handle much more

---

## Strategic Pivot: From Dimensional to Representational Optimization

### The Problem
- **0% SAHI Recall**: Even when ball is "zoomed in" via slicing, model fails → **feature mismatch**, not size
- **Motion Blur Physics**: At 100km/h, ball travels its full diameter (22cm) in 1/120s exposure → appears as **streak**
- **Sim2Real Gap**: Synthetic data (SoccerSynth) lacks sensor noise, compression artifacts, variable lighting
- **Confidence Threshold**: Current 0.5 is too high for small objects (avg confidence ~0.140)

### The Solution
**Pillar 1**: Domain Adaptation (Data-Centric) - **IMMEDIATE PRIORITY**  
**Pillar 2**: Inference Fixes (Threshold & Normalization) - **IMMEDIATE PRIORITY**  
**Pillar 3**: High-Resolution Training (Gradient Accumulation) - **Phase 1.5**  
**Pillar 4**: Architectural Alternative (TrackNet) - **Parallel Development**

---

## Phase 1: Domain Alignment (Epochs 40-45) 🎯 CRITICAL - START IMMEDIATELY

### Action: Enable Aggressive Domain Adaptation Augmentations
**Why**: Addresses the 0% SAHI recall by teaching model to recognize "streaks" and noisy blobs as balls

**Configuration**: Create `configs/resume_with_domain_adaptation.yaml`

**Key Changes**:
1. **Motion Blur**: `prob: 0.5`, `max_kernel_size: 15` (simulates 20-100km/h velocities)
2. **Gaussian Noise**: `prob: 0.3`, `var_limit: [10.0, 50.0]` (sensor noise)
3. **ISO Noise**: `prob: 0.3`, `noise_level: [5, 25]` (stadium lighting conditions)
4. **JPEG Compression**: `prob: 0.2`, `quality_range: [60, 95]` (broadcast compression)
5. **Copy-Paste (Ball-Only)**: `prob: 0.5`, `max_pastes: 3` (class imbalance fix)

**Expected Impact**: +0.03-0.05 improvement (0.598 → 0.63-0.65) by addressing Sim2Real gap

**Risk**: Low - augmentations already implemented in codebase

---

## Phase 1.5: High-Resolution Training (Epochs 45+) 🔍 IMMEDIATE - NOT PHASE 3

### Action: Enable 1288px Resolution with Gradient Accumulation
**Why**: 
- RTX 5090 has 32GB VRAM - **no OOM risk** with proper gradient accumulation
- Resolution is **critical** for <15px objects (current 1120 is too low)
- Gradient accumulation decouples physical batch size from effective batch size

**Configuration**: Update config with:
- `resolution: 1288` (restore original)
- `batch_size: 2` (physical - fits in VRAM)
- `grad_accum_steps: 16` (effective batch = 32, stable for BatchNorm)

**Expected Impact**: +0.02-0.03 improvement (0.63 → 0.65-0.66)

**Risk**: Low - gradient accumulation eliminates OOM concerns

**Note**: This should be **Phase 1.5**, not Phase 3. The "High Risk" assessment was based on misunderstanding of gradient accumulation.

---

## Phase 2: Inference Fixes (Immediate) 🔧 CRITICAL

### Action 1: Lower Confidence Thresholds
**Why**: Average confidence is ~0.140, but threshold is 0.5 → **all detections filtered out**

**Files to Update**:
- `src/perception/local_detector.py`: Change `confidence_threshold: 0.5` → `0.05` (or `0.10`)
- SAHI inference scripts: Ensure threshold is 0.05-0.10 for ball class

**Expected Impact**: Enables detection of valid but low-confidence candidates (tracker can filter noise)

**Risk**: Low - simple parameter change

### Action 2: Normalization Audit
**Why**: Mismatch between training/inference normalization causes 0% SAHI recall

**Action**: 
- Audit `src/perception/local_detector.py` preprocessing
- Ensure bit-exact match with training config (ImageNet mean/std vs 0-1 scaling)
- Check SAHI inference pipeline normalization

**Expected Impact**: Fixes 0% SAHI recall issue

**Risk**: Low - diagnostic and fix

---

## Phase 3: Multi-scale Training (Epochs 50-60) 📐 HIGH PRIORITY

### Action: Enable Multi-scale Training
**Why**: Helps with scale invariance (ball at different distances)

**Configuration**: Use `configs/resume_with_multiscale.yaml`

**Expected Impact**: +0.02-0.03 improvement (0.65 → 0.67-0.68)

**Risk**: Medium - monitor memory, but RTX 5090 should handle it

---

## Phase 4: Architectural Alternative (Parallel Track) 🏗️ LONG-TERM

### Action: Develop TrackNet for Ball Detection
**Why**: 
- Heatmap regression more robust for blurred objects than bounding boxes
- Temporal context (3-frame stack) filters "ghost balls" (socks, lines)
- Specialized architecture for small, fast objects

**Implementation**: 
- Train TrackNet V3/V4 on ball-only dataset
- Run in parallel with RF-DETR (RTX 5090 can handle both)
- Use TrackNet for ball, RF-DETR for players

**Expected Impact**: Significant improvement for fast-moving balls (+0.05-0.10)

**Risk**: Medium - requires new model training

---

## Revised Roadmap Timeline

| Phase | Timeline | Action | Expected Small Objects mAP | Risk |
|-------|----------|--------|----------------------------|------|
| **Phase 1** | Epochs 40-45 | Domain Adaptation Augmentations | 0.63-0.65 | ✅ Low |
| **Phase 1.5** | Epochs 45+ | High-Res (1288px) + Gradient Accum | 0.65-0.66 | ✅ Low |
| **Phase 2** | Immediate | Fix Inference Thresholds | Enables detection | ✅ Low |
| **Phase 3** | Epochs 50-60 | Multi-scale Training | 0.67-0.68 | ⚠️ Medium |
| **Phase 4** | Parallel | TrackNet Development | 0.70-0.75 | ⚠️ Medium |

---

## Implementation Details

### Phase 1: Domain Adaptation Config

Create `configs/resume_with_domain_adaptation.yaml`:

```yaml
# Resume Training - Domain Adaptation for Sim2Real
# Addresses motion blur, sensor noise, compression artifacts

model:
  architecture: detr
  backbone: resnet50
  num_classes: 2
  pretrained: true
  hidden_dim: 256
  nheads: 8
  num_encoder_layers: 6
  num_decoder_layers: 6
  rfdetr_size: base
  remap_mscoco_category: false

training:
  batch_size: 2
  learning_rate: 0.0002
  epochs: 50  # Extended
  weight_decay: 0.0001
  gradient_clip: 0.1
  grad_accum_steps: 20
  resolution: 1120  # Keep for now, increase in Phase 1.5
  num_workers: 1
  device: cuda
  mixed_precision: true
  multi_scale: false  # Enable in Phase 3
  expanded_scales: false

# CRITICAL: Domain Adaptation Augmentations
augmentation:
  train:
    horizontal_flip: 0.5
    
    # Motion Blur - Simulates 20-100km/h ball velocities
    motion_blur:
      enabled: true
      prob: 0.5  # Increased from 0.3
      max_kernel_size: 15  # Simulates different speeds
    
    # Sensor Noise - Simulates DJI Osmo Action 5 ISO grain
    gaussian_blur:
      enabled: true
      prob: 0.3
      kernel_size_range: [3, 7]
      sigma_range: [0.5, 2.0]
    
    iso_noise:
      enabled: true
      prob: 0.3
      noise_level: [5, 25]  # Stadium lighting conditions
    
    # Compression Artifacts - Simulates H.264/H.265 encoding
    jpeg_compression:
      enabled: true
      prob: 0.2
      quality_range: [60, 95]
    
    # Ball-Only Copy-Paste - Addresses class imbalance
    copy_paste:
      enabled: true
      prob: 0.5
      max_pastes: 3  # Artificially increase ball frequency
    
    # Color Jitter - Reduced intensity for small objects
    color_jitter:
      brightness: 0.1  # Reduced from 0.2
      contrast: 0.1
      saturation: 0.1
      hue: 0.05  # Reduced from 0.1

dataset:
  coco_train_path: /workspace/soccer_cv_ball/models/ball_detection_combined_optimized/dataset/train
  coco_val_path: /workspace/soccer_cv_ball/models/ball_detection_combined_optimized/dataset/valid
  category_name: "ball"
  category_id: 0
  ball_class_id: 1
  pin_memory: false
  prefetch_factor: 1
  persistent_workers: false

checkpoint:
  resume_from: /workspace/soccer_cv_ball/models/checkpoint.pth
  start_epoch: 40  # Update after epoch 40
  save_dir: models/checkpoints
```

### Phase 1.5: High-Resolution Config

Create `configs/resume_with_highres_gradaccum.yaml`:

```yaml
# Resume Training - High Resolution with Gradient Accumulation
# Leverages RTX 5090 capabilities without OOM risk

training:
  batch_size: 2  # Physical batch (fits in VRAM)
  grad_accum_steps: 16  # Effective batch = 2 * 16 = 32 (stable)
  resolution: 1288  # RESTORED - critical for <15px objects
  # ... rest same as Phase 1 config
```

### Phase 2: Inference Fixes

**File**: `src/perception/local_detector.py`

**Change**:
```python
# OLD:
def __init__(self, model_path: str, confidence_threshold: float = 0.5, device: str = None):

# NEW:
def __init__(self, model_path: str, confidence_threshold: float = 0.05, device: str = None):
    # For small objects (ball), use lower threshold
    # Tracker will filter false positives using physics (Kalman filter)
```

**SAHI Scripts**: Update all SAHI inference to use `confidence_threshold=0.05` for ball class

---

## Key Differences from Original Strategy

| Aspect | Original Strategy | Revised Strategy |
|--------|------------------|------------------|
| **Phase 1** | Continue current training | **Enable domain adaptation immediately** |
| **Resolution** | Phase 3 (High Risk) | **Phase 1.5 (Low Risk with grad accum)** |
| **Multi-scale** | Phase 2 (Highest Priority) | Phase 3 (After domain adaptation) |
| **Inference** | Not addressed | **Phase 2 (Critical - fixes 0% SAHI)** |
| **Root Cause** | Scale/resolution | **Sim2Real domain gap** |
| **Approach** | Linear progression | **Parallel + Physics-aware** |

---

## Expected Results

### With Revised Strategy:
- **Phase 1** (Domain Adaptation): 0.598 → 0.63-0.65 (+0.03-0.05)
- **Phase 1.5** (High-Res): 0.63 → 0.65-0.66 (+0.02-0.03)
- **Phase 2** (Inference Fixes): Enables detection (fixes 0% SAHI)
- **Phase 3** (Multi-scale): 0.65 → 0.67-0.68 (+0.02-0.03)
- **Total**: 0.598 → **0.67-0.70** ✅ **Target Achieved**

### Why This Works:
1. **Motion Blur**: Teaches model that "streak" = ball (addresses physics)
2. **Noise**: Prevents overfitting to perfect synthetic textures
3. **High-Res**: Preserves <15px object details
4. **Low Threshold**: Enables detection of valid but low-confidence candidates
5. **Gradient Accum**: Enables high-res without OOM (RTX 5090 advantage)

---

## Immediate Action Items

### 1. Create Domain Adaptation Config (Today)
```bash
# Copy base config and add augmentation section
cp configs/resume_20_epochs_low_memory.yaml configs/resume_with_domain_adaptation.yaml
# Then add augmentation section (see above)
```

### 2. Fix Inference Thresholds (Today)
```bash
# Update local_detector.py confidence threshold
# Update SAHI scripts
```

### 3. Start Phase 1 Training (After Epoch 40)
```bash
python scripts/train_ball.py \
    --config configs/resume_with_domain_adaptation.yaml \
    --output-dir models \
    --resume models/checkpoint.pth
```

### 4. Prepare Phase 1.5 (After Epoch 45)
- Create high-res config with gradient accumulation
- Monitor GPU memory to confirm RTX 5090 can handle 1288px

---

## Success Metrics

**Phase 1 Success**: Small objects mAP > 0.63 (from 0.598)  
**Phase 1.5 Success**: Small objects mAP > 0.65  
**Phase 2 Success**: SAHI recall > 0% (currently 0%)  
**Phase 3 Success**: Small objects mAP > 0.67  
**Overall Target**: Small objects mAP ≥ 0.70

---

## Why This Strategy is Better

1. **Addresses Root Cause**: Sim2Real gap, not just scale
2. **Physics-Aware**: Motion blur matches 100km/h ball physics
3. **Hardware-Optimized**: Leverages RTX 5090 capabilities (gradient accum, high-res)
4. **Immediate Impact**: Domain adaptation can start at epoch 40 (no waiting)
5. **Fixes Critical Bug**: 0% SAHI recall addressed in Phase 2
6. **Parallel Development**: TrackNet can be developed alongside RF-DETR improvements

**Conclusion**: The revised strategy transforms the problem from "optimize hyperparameters" to "bridge Sim2Real gap", which is the actual root cause of the 0.598 mAP and 0% SAHI recall.
