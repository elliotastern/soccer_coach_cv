# Ball finetune mix v14 — residual FN after v13 (resume-from-v13)

**Builder:** `scripts/gold_set/build_ball_finetune_v14_residual.py`  
**Pack:** `/Volumes/LaCie/Projects/Soccer project data/ball_finetune_v14_residual/`  
**Manifest:** `data/processed/gold_sets/ball_finetune_v14_residual_manifest.json`  
**Config (Mac):** `configs/finetune_v14_residual_mac.yaml`  
**Resume:** `models/v13_residual_snaps/post_train/checkpoint.pth` @ epoch **264** → absolute end **275** (~10 specialty epochs @560)

Leftover soft-conf / no-det on Match 3 strips scored with **v13** det caches. No new landmarks. No holdout-gallery frames in train.

## Recipe

| Layer | What |
|-------|------|
| Residual FN | `match3_quad_p10_31` / `match3_quad_p8_87` gt frames with focus conf &lt; 0.80 (prefer 0.50–0.79) using `v13_residual_det_cache` |
| Light augs | Hard blur ×2 + tiny paste ×2 on residual items only |
| v12 anchor | Modest `ball_finetune_v12_hard` sample |

## Never train

Match 3 frames **120–194**; holdout gallery clips.

## Gates

Strip `P_emit` ≥ 0.80; holdout clear-ball proxy R ≥ gate (~0.884); do not lower emit / widen agree.

## Build / train (Mac)

```bash
PYTHONPATH=. python scripts/gold_set/build_ball_finetune_v14_residual.py
# then allowlisted MPS train via configs/finetune_v14_residual_mac.yaml
```
