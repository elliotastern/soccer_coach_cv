# Validation Issue Investigation Report

## Summary
Training completed successfully, but validation mAP is 0.0. Investigation reveals the model is predicting all detections as background (label 0).

## Findings

### 1. Training Loss ✅
- **Status**: Training loss decreased properly
- **Evidence**: 
  - Step 50: 1.1697
  - Step 2000: 0.5006
  - Loss decreased from ~1.17 to ~0.50, indicating the model is learning

### 2. Training Logs ✅
- **Status**: No errors found
- **Evidence**: Training completed all 50 epochs without errors
- **Validation runs**: Occurred at epochs 0, 9, 19, 29, 39, 49
- **All validation mAP values**: 0.0000

### 3. Validation Dataset ✅
- **Status**: Dataset is valid
- **Evidence**:
  - 1,012 validation images
  - 5,596 annotations
  - Categories: ['player', 'ball']
  - Sample annotation format is correct

### 4. Evaluation Code ⚠️
- **Status**: Code logic appears correct, but issue identified
- **Problem**: Model predictions are all background (label 0)
- **Evidence from diagnostic**:
  ```
  Image 1: Prediction: 100 boxes, labels: tensor([0])  # ALL background!
  Image 2: Prediction: 100 boxes, labels: tensor([0])  # ALL background!
  ...
  ```

### 5. Root Cause 🔴
**The model is predicting ALL detections as background class (label 0)**

**Diagnostic Results**:
- Model makes predictions (100 boxes per image)
- Prediction scores range: 0.0023 - 0.9980 (reasonable confidence)
- **BUT**: All prediction labels are 0 (background)
- Evaluator skips background predictions (line 67: `if pred_labels[pred_idx] == 0: continue`)
- Result: No predictions match targets → mAP = 0.0

## Possible Causes

1. **Label Mapping Issue**: 
   - DETR outputs labels as 1, 2, 3... (1=first class)
   - Model subtracts 1: `labels = results['labels'] - 1`
   - This should give 0, 1, 2... (0=first class, 1=second class)
   - But all predictions are coming out as 0

2. **Model Not Learning Class Distinction**:
   - Training loss decreased, but model may not have learned to distinguish classes
   - All predictions defaulting to first class (background/class 0)

3. **DETR Post-Processing Issue**:
   - The `post_process_object_detection` might be filtering out non-background classes
   - Or the label mapping in the model wrapper is incorrect

## Recommendations

### Immediate Actions:
1. **Check DETR label output format**: Verify what labels DETR actually outputs
2. **Inspect model predictions**: Add debug logging to see raw DETR outputs before label conversion
3. **Verify training labels**: Ensure training is using correct label mapping (0=player, 1=ball)

### Potential Fixes:
1. **Fix label mapping in `_forward_inference`**:
   - Current: `labels = results['labels'] - 1`
   - May need to handle DETR's label format differently
   - DETR typically outputs: 0=background, 1=first class, 2=second class
   - After `-1`: -1=background (invalid), 0=first class, 1=second class
   - Need to filter background (label 0) BEFORE subtracting

2. **Add confidence threshold filtering**:
   - Filter low-confidence predictions
   - May help if background predictions have lower confidence

3. **Review training label format**:
   - Ensure training uses same label format as inference
   - Check if there's a mismatch between training and inference label handling

## Next Steps

1. **Debug label mapping**: Add logging to see raw DETR outputs
2. **Fix label conversion**: Correct the label mapping in model inference
3. **Re-run validation**: Test with fixed label mapping
4. **If still 0.0**: May need to retrain with corrected label handling

## Files to Review/Modify

- `src/training/model.py` - Line 186: Label conversion logic
- `src/training/evaluator.py` - Line 67: Background filtering logic
- `src/training/dataset.py` - Label mapping for training data
