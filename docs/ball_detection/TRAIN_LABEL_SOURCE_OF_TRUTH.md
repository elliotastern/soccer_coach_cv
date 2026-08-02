# Train label source of truth

**Incident (2026-08-01):** Match finetune used stale `prelabels/annotations.coco.json` while corrected boxes lived only in a newer `annotations.xml`. Training looked like it used “your labels” but did not.

## Rules

1. **Never train from `prelabels/`.** Prelabels are RF-DETR drafts for the editor only.
2. **Corrected packs use `gold/` only:**
   - `gold/annotations.xml` — canonical human-corrected CVAT XML  
   - `gold/annotations.coco.json` — what builders/trainers may read  
3. **After every editor save**, re-export COCO before rebuilding a train pack.
4. **Eval Gold100** (`match1_1_100`) stays out of **train**. It may appear only in the pack `test/` split (strip 0–49 ball frames) from `match1_1_100/gold/`.
5. **COCO is full-res:** editor XML is strip/review (1920×1080); `export_gold_coco.py` scales boxes into `images/` space (3840×2160).

## `math_1_training` layout

```text
data/processed/gold_sets/math_1_training/
├── gold/
│   ├── annotations.xml          # editor load/save
│   └── annotations.coco.json    # train builders
├── images/
├── review/                      # editor points at gold/ XML
└── manifest.json
```

`prelabels/` was removed from this pack on purpose.

## Commands

```bash
# 1) Correct in the viewer (saves gold/annotations.xml)

# 2) Sync COCO from XML
python scripts/gold_set/export_gold_coco.py \
  --gold-dir data/processed/gold_sets/math_1_training \
  --xml data/processed/gold_sets/math_1_training/gold/annotations.xml \
  --output data/processed/gold_sets/math_1_training/gold/annotations.coco.json

# 3) Rebuild RunPod ball mix (fails if gold/ COCO missing)
python3 "/Volumes/LaCie/Projects/Soccer project data/scripts/build_ball_finetune_match_mix.py"
```

## Builder contract

`Soccer project data/scripts/build_ball_finetune_match_mix.py` loads Match balls **only** from:

`math_1_training/gold/annotations.coco.json`

It must not fall back to `prelabels/`.

## New train packs

When generating a fresh CVAT pack:

1. Initial detector draft may still be written under `prelabels/` for first open.
2. As soon as humans correct, **promote** XML+COCO to `gold/`, point the editor at `gold/`, and delete or freeze `prelabels/` so nothing trains from it.
3. Document the pack README with the same `gold/`-only table as `math_1_training`.
