# Match Gold100 — golden test set (player + ball)

**Canonical Phase 1 evaluation set for player and ball detection / classification.**

Use this pack whenever you need a fixed, real-match benchmark for RF-DETR people + ball checkpoints (precision-first). Do not treat SoccerSynth or random clips as a substitute for this set once human correction is complete.

**Do not train on this pack.** Match train labels live in `math_1_training` under `gold/` only — see [`docs/ball_detection/TRAIN_LABEL_SOURCE_OF_TRUTH.md`](../ball_detection/TRAIN_LABEL_SOURCE_OF_TRUTH.md).

| | |
|---|---|
| **Pack path** | `data/processed/gold_sets/match1_1_100/` (images local-only; **corrected XML + manifest are tracked in git**) |
| **Classes** | `player` (COCO category 1), `ball` (COCO category 2) |
| **Size** | 100 stratified frames from Match 1 multi-cam |
| **Source video** | `data/raw/Match 1/Match 1 -1` (cams 8 / 9 / 11 / 13, 3840×2160) |
| **Review UI** | http://localhost:8080/gold100 → `annotation/gold100_editor.html` |

## What it measures

- **Player / person detection** — boxes labeled `player` (do not label coaches/referees as players unless you intentionally add those classes).
- **Ball detection** — boxes labeled `ball`.
- Metrics from `scripts/gold_set/eval_on_gold100.py`: precision / recall at IoU 0.5 and 0.8, with the Phase 1 high-precision bar (drop low-confidence predictions).

It is a **detection / classification** gold set on sparse frames. It is **not** a tracking or Kalman/ByteTrack gold set (that needs continuous clips).

## Pack contents (after build + harden)

```text
data/processed/gold_sets/match1_1_100/
├── images/                 # full-res JPGs (gold COCO image space)
├── gold/
│   ├── annotations.xml     # canonical corrected CVAT XML (editor load/save)
│   └── annotations.coco.json  # full-res boxes (strip XML scaled ×2)
├── review/
│   ├── frames/000.jpg…     # 1920×1080 review sequence (editor display)
│   ├── strip_100.mp4       # all-intra H.264 (secondary)
│   └── editor.html         # generated image-sequence editor
├── manifest.json
└── README.md
```

After human correction, export COCO from `gold/`:

```bash
python scripts/gold_set/export_gold_coco.py \
  --gold-dir data/processed/gold_sets/match1_1_100 \
  --xml data/processed/gold_sets/match1_1_100/gold/annotations.xml \
  --output data/processed/gold_sets/match1_1_100/gold/annotations.coco.json
```

## Workflow

1. **Build** (once, or when source/checkpoints change):

   ```bash
   python scripts/gold_set/build_match_gold100.py
   ```

2. **Harden review pack** (image-sequence editor + alignment checks):

   ```bash
   python scripts/gold_set/harden_review_pack.py --gold-dir data/processed/gold_sets/match1_1_100
   ```

3. **Correct labels**

   ```bash
   python serve_viewer.py
   # open http://localhost:8080/gold100
   ```

   - Scrub with the frame slider (JPG sequence — boxes stay aligned).
   - Click-drag a box to move; drag corners to resize; `N` then drag to add; Save writes XML.

4. **Export + evaluate**

   ```bash
   python scripts/gold_set/export_gold_coco.py --gold-dir data/processed/gold_sets/match1_1_100
   python scripts/gold_set/eval_on_gold100.py --gold-dir data/processed/gold_sets/match1_1_100
   ```

5. **Editor smoke test** (server must be on port 8080):

   ```bash
   python scripts/gold_set/test_gold100_editor.py
   ```

## Scripts

| Script | Role |
|--------|------|
| `scripts/gold_set/build_match_gold100.py` | Sample 100 strata, RF-DETR prelabel, write images / COCO / CVAT XML |
| `scripts/gold_set/harden_review_pack.py` | Review JPGs, all-intra strip, generate `gold100_editor.html` |
| `scripts/gold_set/export_gold_coco.py` | Corrected XML → full-res COCO gold |
| `scripts/gold_set/eval_on_gold100.py` | Player + ball P/R vs gold COCO |
| `scripts/gold_set/test_gold100_editor.py` | Playwright: seek, move, resize, draw, save |

## Policy

- Prefer commercial-use-safe Match footage only.
- High precision, lower recall: confidence below ~80% → drop / do nothing in product thresholds (eval may still report at other cutoffs).
- After correction, treat `annotations.coco.json` from export as the frozen GT for model comparisons; note checkpoint paths and date in any report under `reports/`.

## Results so far

- **Frames 0–20 labelled** (2026-07-31): player P≈97% / R≈35% at IoU 0.5, conf ≥ 0.8 (**precision PASS**); ball **0** dets at conf ≥ 0.8. Full table: [`reports/gold100_frames0_20_eval.md`](../../reports/gold100_frames0_20_eval.md).
- **Ball prelabel stack ablation** (SAHI / size / multiscale / Kalman): [`reports/gold100_ball_prelabel_stack.md`](../../reports/gold100_ball_prelabel_stack.md).  
  Recommended prelabel: `thr=0.30 + size filter + topk=2` (`LocalRFDETRDetector(enhance_ball=True)` or `BallPrelabeler`). SAHI/Kalman validated **off** for sparse Match gold with `ball_89.pth` (need domain finetune for product-level ball).
- **Feature scorecard (≥9/10 each):** [`reports/feature_scorecard.md`](../../reports/feature_scorecard.md) — run `python scripts/gold_set/test_feature_scorecard.py`.
