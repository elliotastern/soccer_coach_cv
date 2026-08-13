# Match 2 train label 100 (best)

Best-ranked large-ball proposals for **train** labeling (eval gold stays separate).

| | |
|--|--|
| Frames | 100 |
| Cams | {'Cam5plus': 40, 'Cam4plus': 39, 'P10': 21} |
| Mean conf | 0.653 |
| Mean side | 50 px |

## Label (local editor — same as accepted50)

http://127.0.0.1:8080/match2-train100

Ball → **N** → draw · **Save** writes `gold/annotations.xml` (source of truth).

Rebuild train/test: `python3 scripts/gold_set/build_match2_train_test_split.py`

## CVAT

Import `cvat/images/` + `cvat/annotations.xml` (CVAT for images 1.1).
