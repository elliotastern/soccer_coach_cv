# Gold100 (`match1_1_100`) — eval only

Canonical Match-1 eval set. **Do not train on this pack.**

## Source of truth

| Path | Role |
|---|---|
| `gold/annotations.xml` | Canonical corrected CVAT XML (editor load/save) |
| `gold/annotations.coco.json` | Full-res COCO (strip XML scaled 1920→3840) |
| `images/` | Full-res frames |
| `review/` | Strip editor + frames |

`prelabels/` was archived after promotion (`prelabels_archived_*`).

## Review

```bash
python serve_viewer.py
# http://127.0.0.1:8765/data/processed/gold_sets/match1_1_100/review/editor.html
```

Save writes `gold/annotations.xml`. Re-export COCO after edits:

```bash
python scripts/gold_set/export_gold_coco.py \
  --gold-dir data/processed/gold_sets/match1_1_100 \
  --xml data/processed/gold_sets/match1_1_100/gold/annotations.xml \
  --output data/processed/gold_sets/match1_1_100/gold/annotations.coco.json
```

Test split in `ball_finetune_match_mix` uses strip **0–49** frames with ≥1 ball.
