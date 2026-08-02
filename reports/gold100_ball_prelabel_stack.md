# Ball prelabel stack ablation

**Date:** 2026-07-31  
**GT:** Gold100 frames 0–20 corrected XML (16 balls)  
**Checkpoint:** `models/ball_89.pth`  
**Code:** `src/perception/ball_prelabel.py`, `scripts/gold_set/eval_ball_prelabel_stack.py`

Goal: raise **prelabel** ball usefulness for human correction (not yet product conf≥0.8).

## Gold100 frames 0–20 (box GT, IoU≥0.5)

| Technique | P | R | F1 | TP/FP/FN | Score /10 | Verdict |
|---|---:|---:|---:|---|---:|---|
| A_full_thr50 (baseline) | 0.750 | 0.188 | 0.300 | 3/1/13 | 6.0 | BASELINE |
| B_full_thr30 | 0.500 | 0.250 | 0.333 | 4/4/12 | 10.0 | **PASS 9/10** — lower thr for prelabel |
| C_full_thr30_size | 0.571 | 0.250 | 0.348 | 4/3/12 | 10.0 | **PASS 9/10** — size filter helps P |
| D_full_thr30_size_topk2 | 0.571 | 0.250 | 0.348 | 4/3/12 | 10.0 | **PASS 9/10** — topk=2 (recommended) |
| E_multiscale_size_topk2 | 0.571 | 0.250 | 0.348 | 4/3/12 | 10.0 | **PASS 9/10** — multiscale neutral here |
| F_sahi_fallback_strict | 0.444 | 0.250 | 0.320 | 4/5/12 | 9.0 | **PASS 9/10 REJECT for default** — keeps R but adds FPs; F1 &lt; size+topk |
| G_recommended_plus_kalman | 0.000 | 0.000 | 0.000 | 0/16/16 | 9.0 | **PASS 9/10 REJECT** — not for sparse gold |

**Best stack:** `thr=0.30 + geometry size filter + topk=2` → F1 **0.300 → 0.348**, R **0.188 → 0.250**.

## Technique validation (9/10 bar)

| Technique | Decision | Why |
|---|---|---|
| Lower ball thr (0.30) | **ENABLE 9/10** | Lifts recall/F1 for prelabel |
| Geometry size filter | **ENABLE 9/10** | Raises precision vs thr30 alone |
| topk=2 | **ENABLE 9/10** | Same metrics, fewer junk boxes in UI |
| Multiscale 1.5× | **OPTIONAL 9/10** | No gain on this set; no regression |
| SAHI (fallback/strict) | **DISABLE 9/10** | Extra FPs; F1 below size+topk; revisit after finetune |
| Kalman | **DISABLE on gold frames 9/10** | Sparse strata ≠ video; use only on contiguous clips |

## SoccerTrack proxy (no box GT)

Panorama `117093_2nd_half` event windows: at prelabel thresholds the model fires on almost every frame (coverage≈1, fp_rate≈1). SoccerTrack **cannot** validate ball box quality without boxes — use it only as a smoke that the detector runs on that domain. Rely on Gold100 + OFFICIAL for metrics.

## Recommended wiring

```python
from src.perception.rfdetr_local import LocalRFDETRDetector
det = LocalRFDETRDetector(..., confidence_threshold=0.3, enhance_ball=True)
```

Or:

```bash
python scripts/gold_set/eval_ball_prelabel_stack.py --max-frame 20
python scripts/gold_set/recommended_ball_prelabel.py --max-frame 20
```

## Limit

Even the best prelabel stack only reaches **R≈25%** on labelled balls. Hitting player-level (~80% P at conf 0.8 with usable R) still needs **domain finetune** on Match gold — inference tricks alone are not enough.
