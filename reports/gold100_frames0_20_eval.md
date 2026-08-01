# Gold100 frames 0–20 — player & ball eval vs Phase 1

**Date:** 2026-07-31  
**GT:** corrected `data/processed/gold_sets/match1_1_100/prelabels/annotations.xml` (frames 0–20 labelled)  
**Images:** `review/frames/000.jpg`…`020.jpg` (1920×1080; same coords as XML)  
**Detector:** `LocalRFDETRDetector`  
- people: `models/people_after_100_epochs.pth`  
- ball: `models/ball_89.pth`  

**GT counts (frames 0–20):** 278 players, 16 balls

## Phase 1 goals (from product scope)

- Prefer **high precision** over recall  
- Confidence below ~**80%** → drop / do nothing  
- Functional target ~**80%** (precision-first at the operating bar)

## Primary operating point — IoU ≥ 0.5, conf ≥ 0.8

| Class | Precision | Recall | TP | FP | FN | vs Phase 1 |
|-------|-----------|--------|----|----|----|------------|
| player | **97.0%** | 35.3% | 98 | 3 | 180 | Precision **PASS** (≥80%); recall low (acceptable if precision-first) |
| ball | **0.0%** | 0.0% | 0 | 0 | 16 | **FAIL** — no ball detections above 0.8 |

## Looser cutoff — IoU ≥ 0.5, conf ≥ 0.5

| Class | Precision | Recall | TP | FP | FN |
|-------|-----------|--------|----|----|----|
| player | 93.7% | 86.0% | 239 | 16 | 39 |
| ball | 75.0% | 18.8% | 3 | 1 | 13 |

## Strict boxes — IoU ≥ 0.8, conf ≥ 0.8

| Class | Precision | Recall | TP | FP | FN |
|-------|-----------|--------|----|----|----|
| player | 92.1% | 33.5% | 93 | 8 | 185 |
| ball | 0.0% | 0.0% | 0 | 0 | 16 |

## Takeaway

On labelled frames 0–20, **players meet the Phase 1 precision goal** at conf ≥ 0.8 (P≈97%) with low recall. **Ball does not** — at the product threshold it predicts nothing; even at conf ≥ 0.5 it only recovers ~19% of balls (P≈75%).

## Reproduce

```bash
# Requires corrected XML for frames 0–20 and review frames present.
# Eval was run with LocalRFDETR on review/frames vs XML strip coords.
python scripts/gold_set/eval_on_gold100.py --gold-dir data/processed/gold_sets/match1_1_100
```

Note: full-pack `eval_on_gold100.py` uses COCO on all images; this report is the frames **0–20** slice against corrected XML only.
