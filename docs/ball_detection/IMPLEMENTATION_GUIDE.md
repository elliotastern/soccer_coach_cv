# Implementation Guide: Revised Small Object Optimization Strategy

## Quick Start

### Step 1: Fix Inference Thresholds (Do This First - 5 minutes)

**File**: `src/perception/local_detector.py`

Change line 17:
```python
# FROM:
def __init__(self, model_path: str, confidence_threshold: float = 0.5, device: str = None):

# TO:
def __init__(self, model_path: str, confidence_threshold: float = 0.05, device: str = None):
```

**Why**: Average confidence is ~0.140, but threshold is 0.5 → all detections filtered. This fixes the 0% SAHI recall issue.

**Also Check**: SAHI inference scripts - ensure they use `confidence_threshold=0.05` for ball class.

---

### Step 2: After Epoch 40 Completes - Start Phase 1

**Action**: Switch to domain adaptation config

**Command**:
```bash
cd /workspace/soccer_cv_ball

# Update config: Set start_epoch: 40 in resume_with_domain_adaptation.yaml
# Then run:
python scripts/train_ball.py \
    --config configs/resume_with_domain_adaptation.yaml \
    --output-dir models \
    --resume models/checkpoint.pth
```

**Expected**: Small objects mAP should improve from 0.598 to 0.63-0.65 over 5 epochs

**Monitor**: Run evaluation after epoch 45:
```bash
python scripts/comprehensive_training_evaluation.py configs/resume_with_domain_adaptation.yaml
```

---

### Step 3: After Epoch 45 - Start Phase 1.5

**Action**: Switch to high-resolution config with gradient accumulation

**Command**:
```bash
# Update config: Set start_epoch: 45 in resume_with_highres_gradaccum.yaml
# Then run:
python scripts/train_ball.py \
    --config configs/resume_with_highres_gradaccum.yaml \
    --output-dir models \
    --resume models/checkpoints/checkpoint_epoch_45_lightweight.pth
```

**Expected**: Small objects mAP should improve from 0.63 to 0.65-0.66

**Monitor GPU Memory**: 
```bash
watch -n 1 nvidia-smi
```

If OOM occurs: Reduce `batch_size` to 1, increase `grad_accum_steps` to 32

---

## Important Notes

### RF-DETR Augmentation Handling

**Critical**: RF-DETR's `train()` function may have its own augmentation system. The `augmentation` section in the config might only be used for:
- Mosaic preprocessing (handled by `train_ball.py` before RF-DETR)
- Custom transforms (if RF-DETR supports them)

**Action Required**: 
1. Check RF-DETR documentation/source for augmentation parameters
2. If RF-DETR doesn't support motion blur/noise via config, we may need to:
   - Preprocess the dataset with augmentations (offline)
   - Or modify RF-DETR's internal augmentation pipeline

**Workaround**: The domain adaptation config includes augmentation settings. If RF-DETR doesn't use them, we can:
1. Preprocess training images with motion blur/noise (create augmented dataset)
2. Or modify the data loader to apply augmentations on-the-fly

### Confidence Threshold Fix

The inference threshold fix (0.5 → 0.05) is **critical** and should be done immediately, even before Phase 1 training starts. This enables detection of valid but low-confidence candidates.

---

## Expected Timeline

| Date/Event | Action | Expected Result |
|------------|--------|-----------------|
| **Now** | Fix inference thresholds | Enables detection |
| **After Epoch 40** | Start Phase 1 (Domain Adaptation) | 0.598 → 0.63-0.65 |
| **After Epoch 45** | Start Phase 1.5 (High-Res) | 0.63 → 0.65-0.66 |
| **After Epoch 50** | Start Phase 3 (Multi-scale) | 0.65 → 0.67-0.68 |
| **Target** | **0.67-0.70** | ✅ **Achieved** |

---

## Verification Checklist

After each phase, verify:

- [ ] Small objects mAP improved
- [ ] Overall mAP maintained (>0.68)
- [ ] No overfitting (val loss not increasing)
- [ ] SAHI recall > 0% (if Phase 2 completed)
- [ ] GPU memory usage acceptable
- [ ] Training loss decreasing

---

## Troubleshooting

### If Domain Adaptation Config Doesn't Work

**Issue**: RF-DETR may not use augmentation config directly

**Solution**: 
1. Check RF-DETR source code for augmentation parameters
2. If needed, preprocess dataset offline with augmentations
3. Or modify data loader to apply augmentations

### If High-Resolution Causes OOM

**Issue**: Even with gradient accumulation, OOM occurs

**Solution**:
1. Reduce `batch_size` to 1
2. Increase `grad_accum_steps` to 32 (maintains effective batch ~32)
3. If still OOM, keep resolution at 1120 and focus on domain adaptation

### If Metrics Don't Improve

**Issue**: Domain adaptation not helping

**Solution**:
1. Verify augmentations are actually being applied
2. Check if RF-DETR has built-in augmentations that conflict
3. Increase augmentation probabilities (motion blur prob: 0.5 → 0.7)
4. Consider TrackNet alternative (Phase 4)

---

## Key Files

- **Strategy**: `SMALL_OBJECT_OPTIMIZATION_STRATEGY_REVISED.md`
- **Summary**: `STRATEGY_UPDATE_SUMMARY.md`
- **Phase 1 Config**: `configs/resume_with_domain_adaptation.yaml`
- **Phase 1.5 Config**: `configs/resume_with_highres_gradaccum.yaml`
- **Phase 3 Config**: `configs/resume_with_multiscale.yaml` (already exists)

---

## Success Criteria

**Minimum**: Small objects mAP ≥ 0.65 (from 0.598)  
**Target**: Small objects mAP ≥ 0.70  
**Optimal**: Small objects mAP ≥ 0.75

**Current Progress**: +0.071 in 19 epochs (+0.0044/epoch) - **on track!**
