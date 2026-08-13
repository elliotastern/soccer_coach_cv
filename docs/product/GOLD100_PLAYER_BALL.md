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
- Metrics from `scripts/gold_set/eval_on_gold100.py`: precision / recall at IoU 0.5 for conf cutoffs **≥ 0.5 and ≥ 0.8** (product emit bar uses ≥ 0.8).

**PoC pass for ball (Phase 1):** precision of **emitted** predictions at conf ≥ 0.8, IoU 0.5 **≥ 0.80**. That is the client “~80% accuracy” acceptance reading — see [PHASE1_SCOPE.md](PHASE1_SCOPE.md).

**Engineering diagnostics (not the sole client bar):** recall @0.5 / @0.8 (R50/R80) and **clear-ball recall** (near/clear views; e.g. min side ≥ 25 px on full-res). Use these to track coverage progress.

Gold100 is **one camera per frame** (stratified across cams 8/9/11/13). Multi-cam selection / system metrics on synced clips are complementary and do not replace this pack.

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
| `scripts/gold_set/eval_poc_ball_metrics.py` | Ball PoC: P_emit @0.8 + clear-ball recall (strip 0–49; optional `--require-ball-gt`) |
| `scripts/gold_set/select_checkpoint_by_gold_poc.py` | Rank multiple `.pth` by Gold PoC |
| `scripts/gold_set/eval_system_ball_poc.py` | Product stack: SAHI + Kalman + track emit + multi-cam selection on multicam 20s |
| `scripts/gold_set/wait_v8_then_poc.sh` | Wait for remote train, pull ckpts, run PoC select |
| `scripts/gold_set/test_gold100_editor.py` | Playwright: seek, move, resize, draw, save |

## Policy

- Prefer commercial-use-safe Match footage only.
- High precision, lower recall: confidence below ~80% → drop / do nothing in product thresholds (eval may still report at other cutoffs).
- Do not treat all-frame R@0.8 ≈ 0.8 as the Phase 1 PoC gate; PoC pass is **P_emit ≥ 0.80** at conf ≥ 0.8.
- After correction, treat `annotations.coco.json` from export as the frozen GT for model comparisons; note checkpoint paths and date in any report under `reports/`.
- **Train checkpoint selection:** pack COCO val each epoch is diagnostics only. Rank ball finetunes on **Gold strip 0–49** via PoC metrics (`P_emit` + clear-ball recall), not pack AP50. Full policy: [`docs/ball_detection/TRAIN_CHECKPOINT_SELECTION.md`](../ball_detection/TRAIN_CHECKPOINT_SELECTION.md). Do not interrupt a healthy mid-run train to rewire eval—run offline PoC at end (and every **N=5** epochs on future jobs).
- **Product system bar:** after checkpoint pick, score **emits** with track gate + SAHI (+ Kalman) + multi-cam selection: `python scripts/gold_set/eval_system_ball_poc.py`. Report **n_emitted**; hollow P_emit with tiny n is not a pass.

## Results so far

- **Frames 0–20 labelled** (2026-07-31): player P≈97% / R≈35% at IoU 0.5, conf ≥ 0.8 (**precision PASS**); ball **0** dets at conf ≥ 0.8. Full table: [`reports/gold100_frames0_20_eval.md`](../../reports/gold100_frames0_20_eval.md).
- **Ball prelabel stack ablation** (SAHI / size / multiscale / Kalman): [`reports/gold100_ball_prelabel_stack.md`](../../reports/gold100_ball_prelabel_stack.md).  
  Recommended prelabel: `thr=0.30 + size filter + topk=2` (`LocalRFDETRDetector(enhance_ball=True)` or `BallPrelabeler`). SAHI/Kalman validated **off** for sparse Match gold with `ball_89.pth` (need domain finetune for product-level ball).
- **Feature scorecard (≥9/10 each):** [`reports/feature_scorecard.md`](../../reports/feature_scorecard.md) — run `python scripts/gold_set/test_feature_scorecard.py`.
