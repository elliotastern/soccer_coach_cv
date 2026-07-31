# Training Plan: Epochs 35 → 60

## Phase 1: Complete to Epoch 40 ✅
**Status:** Ready to start
**Config:** `configs/resume_20_epochs_low_memory.yaml`
- Current: Epoch 35
- Target: Epoch 40
- Remaining: 5 epochs

## Phase 2: Evaluate at Epoch 40
**Action:** Run evaluation script
```bash
python scripts/evaluate_training_progress.py configs/resume_20_epochs_low_memory.yaml
```

**Check:**
- [ ] Is mAP still improving?
- [ ] Is training loss decreasing?
- [ ] Any overfitting signs?
- [ ] Small objects (ball) mAP progress

## Phase 3: Continue to 50 Epochs (if improving)
**Config:** `configs/resume_to_50_epochs.yaml`
- Start: Epoch 40
- Target: Epoch 50
- Remaining: 10 epochs

**Update checkpoint path** in config after epoch 40 completes.

## Phase 4: Evaluate at Epoch 50
**Action:** Run evaluation script
```bash
python scripts/evaluate_training_progress.py configs/resume_to_50_epochs.yaml
```

## Phase 5: Continue to 60 Epochs (if still improving)
**Config:** `configs/resume_to_60_epochs.yaml`
- Start: Epoch 50
- Target: Epoch 60
- Remaining: 10 epochs

**Update checkpoint path** in config after epoch 50 completes.

## Monitoring Checklist

### After Each Phase:
- [ ] Check validation mAP@0.5:0.95
- [ ] Check small objects mAP (ball detection)
- [ ] Compare train vs validation loss
- [ ] Look for overfitting signs
- [ ] Update checkpoint path in next config

### Stop Training If:
- ❌ Validation loss increasing while train loss decreases
- ❌ Metrics plateauing for 3+ epochs
- ❌ Clear overfitting signs
- ✅ Target metrics achieved

### Continue Training If:
- ✅ Metrics still improving
- ✅ Small objects mAP < 0.70 (room for improvement)
- ✅ No overfitting signs
- ✅ Training loss still decreasing

## Quick Commands

### Resume Training:
```bash
cd /workspace/soccer_cv_ball
# Use your training script with appropriate config
```

### Evaluate Progress:
```bash
python scripts/evaluate_training_progress.py <config_file>
```

### Check Metrics:
```bash
tail -50 training_resume.log | grep "Average Precision"
```
