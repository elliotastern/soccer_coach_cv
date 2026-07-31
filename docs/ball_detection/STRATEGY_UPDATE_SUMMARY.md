# Strategy Update Summary: Physics-Aware Small Object Optimization

## Key Changes from Original Strategy

### Critical Insight
The **0% SAHI recall** indicates a **Sim2Real domain gap**, not just scale/resolution. The original strategy focused on dimensional optimization (size, scale) but missed the representational problem (feature mismatch).

### Root Cause Analysis
1. **Motion Blur Physics**: Balls at 100km/h appear as **streaks** (22cm travel in 1/120s exposure), not spheres
2. **Sim2Real Gap**: Synthetic data (SoccerSynth) lacks sensor noise, compression artifacts, variable lighting
3. **Confidence Threshold**: Current 0.5 is too high (avg confidence ~0.140) → all detections filtered
4. **Underutilized Hardware**: RTX 5090 (32GB VRAM) can handle much more than current config

## Revised Strategy Phases

### Phase 1: Domain Alignment (Epochs 40-45) - **IMMEDIATE**
**Action**: Enable aggressive domain adaptation augmentations
- Motion blur (prob=0.5, simulates 20-100km/h velocities)
- Sensor noise (Gaussian, ISO)
- Compression artifacts (JPEG)
- Ball-only copy-paste (class imbalance)

**Expected**: 0.598 → 0.63-0.65 (+0.03-0.05)
**Config**: `configs/resume_with_domain_adaptation.yaml`

### Phase 1.5: High-Resolution Training (Epochs 45+) - **IMMEDIATE**
**Action**: Restore 1288px resolution with gradient accumulation
- Physical batch: 2 (fits in VRAM)
- Gradient accumulation: 16 steps
- Effective batch: 32 (stable for BatchNorm)

**Expected**: 0.63 → 0.65-0.66 (+0.02-0.03)
**Config**: `configs/resume_with_highres_gradaccum.yaml`
**Risk**: Low (gradient accumulation eliminates OOM)

### Phase 2: Inference Fixes - **IMMEDIATE**
**Action**: Lower confidence thresholds and fix normalization
- Change threshold: 0.5 → 0.05 (for ball class)
- Audit normalization mismatch (training vs inference)
- Fix SAHI inference pipeline

**Expected**: Enables detection (fixes 0% SAHI recall)
**Files**: `src/perception/local_detector.py`, SAHI scripts

### Phase 3: Multi-scale Training (Epochs 50-60)
**Action**: Enable multi-scale training
**Expected**: 0.65 → 0.67-0.68 (+0.02-0.03)
**Config**: `configs/resume_with_multiscale.yaml`

### Phase 4: TrackNet Alternative (Parallel)
**Action**: Develop TrackNet for ball detection (heatmap + temporal)
**Expected**: 0.67 → 0.70-0.75 (+0.03-0.08)

## Comparison: Original vs Revised

| Aspect | Original | Revised |
|--------|----------|---------|
| **Phase 1** | Continue current training | **Domain adaptation immediately** |
| **Resolution** | Phase 3 (High Risk) | **Phase 1.5 (Low Risk)** |
| **Multi-scale** | Phase 2 (Highest) | Phase 3 (After domain adapt) |
| **Inference** | Not addressed | **Phase 2 (Critical)** |
| **Root Cause** | Scale/resolution | **Sim2Real gap** |
| **Approach** | Linear | **Parallel + Physics-aware** |

## Expected Results

**Original Strategy**: 0.598 → 0.67-0.69 (over 3 phases, ~20 epochs)
**Revised Strategy**: 0.598 → 0.67-0.70 (over 3 phases, ~20 epochs, but addresses root cause)

**Key Difference**: Revised strategy fixes the **0% SAHI recall** issue and addresses the physics of high-velocity ball detection.

## Immediate Actions

1. ✅ Create `configs/resume_with_domain_adaptation.yaml` (DONE)
2. ✅ Create `configs/resume_with_highres_gradaccum.yaml` (DONE)
3. ⏳ Update `src/perception/local_detector.py` confidence threshold (0.5 → 0.05)
4. ⏳ Audit normalization in inference pipeline
5. ⏳ Start Phase 1 training after epoch 40 completes

## Files Created

1. `SMALL_OBJECT_OPTIMIZATION_STRATEGY_REVISED.md` - Full revised strategy
2. `configs/resume_with_domain_adaptation.yaml` - Phase 1 config
3. `configs/resume_with_highres_gradaccum.yaml` - Phase 1.5 config
4. `STRATEGY_UPDATE_SUMMARY.md` - This summary

## Next Steps

1. **After Epoch 40**: Switch to domain adaptation config (Phase 1)
2. **After Epoch 45**: Switch to high-res config (Phase 1.5)
3. **Immediately**: Fix inference confidence thresholds (Phase 2)
4. **After Epoch 50**: Enable multi-scale (Phase 3)
