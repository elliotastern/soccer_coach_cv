# Match 1 CVAT train pack (`math_1_training`)

Fresh **100-frame random** Match-1 sample for ball/player correction.  
**Not** the Gold100 eval set (`match1_1_100`). Sampling excludes Gold100 images.

Source: `data/raw/Match 1/Match 1 -1`

## Source of truth (corrected labels)

| Path | Role |
|---|---|
| `gold/annotations.xml` | **Canonical** corrected CVAT XML (editor load/save) |
| `gold/annotations.coco.json` | Export for training packs — rebuild after every XML save |
| `images/` | Full-res frames |
| `review/` | Strip editor + frames |

There is **no** `prelabels/` folder on this pack anymore. Stale RF-DETR prelabel COCO was deleted so training cannot silently use old boxes.

## Review / correct

```bash
python serve_viewer.py
```

Open the math_1_training editor. Save writes:

`data/processed/gold_sets/math_1_training/gold/annotations.xml`

## After every correction → re-export COCO

```bash
python scripts/gold_set/export_gold_coco.py \
  --gold-dir data/processed/gold_sets/math_1_training \
  --xml data/processed/gold_sets/math_1_training/gold/annotations.xml \
  --output data/processed/gold_sets/math_1_training/gold/annotations.coco.json
```

Then rebuild the RunPod mix (reads **`gold/` only**):

```bash
python3 "/Volumes/LaCie/Projects/Soccer project data/scripts/build_ball_finetune_match_mix.py"
```

See: [`docs/ball_detection/TRAIN_LABEL_SOURCE_OF_TRUTH.md`](../../../../docs/ball_detection/TRAIN_LABEL_SOURCE_OF_TRUTH.md)
