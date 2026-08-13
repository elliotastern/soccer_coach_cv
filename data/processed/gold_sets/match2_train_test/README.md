# Match 2 train / test

Source of truth: `gold/annotations.xml` (not `prelabels/`).

| Split | Pack | Frames | Ball boxes |
|-------|------|--------|------------|
| train | `match2_train_label100` | 87 | 124 |
| valid | last 10 of labeled train100 | 10 | 10 |
| test | `match2_gold_frames` (held out) | 50 | 62 |

Skipped unlabeled train frames: 3.

RF-DETR layout:

```
data/processed/gold_sets/match2_train_test/{train,valid,test}/_annotations.coco.json
```
