# Ball finetune mix v13 — residual FN (resume-from-v12)

**Builder:** `scripts/gold_set/build_ball_finetune_v13_residual.py`  
**Pack:** `/Volumes/LaCie/Projects/Soccer project data/ball_finetune_v13_residual/`  
**Manifest:** `data/processed/gold_sets/ball_finetune_v13_residual_manifest.json`  
**Config:** `configs/finetune_v13_residual.yaml` (Catch: `*_catch.yaml`, Mac fallback: `*_mac.yaml`)  
**Resume:** `models/v12_hard_snaps/post_train/checkpoint.pth` @ epoch **254** → absolute end epoch **315** (~60 specialty epochs; RF-DETR resume uses absolute epoch counts)

Thin **specialty** after Phase 0 detector+hull triage: lift Match 3 strip soft-conf / no-det misses without redoing the full v12 mix.

## Recipe

| Layer | What |
|-------|------|
| Residual FN | Clear frames on `match3_quad_p10_31` / `match3_quad_p8_87` (P9 if present) with gold `gt_balls`, focus det conf **&lt; 0.80** (prefer **0.50–0.79**, also **0.20–0.49** / low / **no_det**) from strip `det_cache` |
| Light augs | Hard blur ×2 + tiny paste ×2 on **residual items only** |
| v12 anchor | Modest sample of existing `ball_finetune_v12_hard` train/valid COCO images so medium balls do not collapse |
| `bbox_tight` | Optional 24 px center squares recorded in the manifest; **primary COCO boxes stay normal gold xywh** |

## Never train

Same Match 3 holdout as v11: frames **120–194** stay out of train (honest M1 strip / clear-ball R).

## Gates (after train)

| Gate | Target |
|------|--------|
| Strip `P_emit` | ≥ **0.80** (`score_match3_ball_m1.py` on P10 + P8 packs) |
| Holdout clear-ball R | Product fuse (F0); do not lower emit / widen agree |

## Paths

| Env | Dataset / checkpoint |
|-----|----------------------|
| RunPod (config default) | `/workspace/soccer_cv_ball/datasets/ball_finetune_v13_residual/` · resume `…/models/v12_hard_snaps/post_train/checkpoint.pth` |
| Catch | `/home/catch/soccer_coach_cv/` (or sync pack under a `datasets/` mirror); weights `~/soccer_coach_cv/models/v12_hard_snaps/post_train/checkpoint.pth` |
| Mac build output | `/Volumes/LaCie/Projects/Soccer project data/ball_finetune_v13_residual/` |

## Build

```bash
python3 scripts/gold_set/build_ball_finetune_v13_residual.py
python3 scripts/gold_set/test_build_ball_finetune_v13_residual.py
```

## Train / A/B status

- Pack built (810 train / 39 valid; 122 residual FN frames P8+P10). Landmarks/L2 closed (FOV fully labelled).
- Catch/RunPod unreachable; Mac MPS specialty **10 epochs @560** (255→264) → `models/v13_residual_snaps/post_train/checkpoint.pth`.
- Strip A/B vs v12_hard (`ab_v13_residual_vs_v12.json`): P10 clear_R **0.879→0.888**, P8 **0.846→0.904**, P_emit **≥0.99** → **promoted** into `configs/default.yaml` (v12_hard kept on disk).
- Optional later: Catch @1288 resume for emit-safe resolution parity; re-score holdout galleries with new det caches.