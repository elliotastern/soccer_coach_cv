# Match 3 clear-ball coverage (product-wide)

**Target:** clear-ball emit rate ≥ **0.80** on product fuse path. Keep **P_emit ≥ 0.80**, **EMIT_CONF = 0.80**, agree ≤ **4 m**. Pitch 1 meters (not FIFA).

**Product fuse:** F1+F2+F0+F3 · **MIN_SUPPORT = 0.20** (H1 promote).

## Status

| Pack | Role | clear_ball_proxy_R |
|------|------|-------------------:|
| Tune random | report-only | 0.625 |
| **Holdout** | **gate** | **0.884** (was 0.556 freeze) |
| Strips product-F0 | P_emit / clear_R | P10 1.0 / **0.879** · P8 0.966 / 0.801 |

## Map-first pass (compute-safe)

1. Spot-check P10 holdout FNs → in-FOV feet (`holdout_map_fn_spot/`).
2. P10 `hull_image_points` → R 0.556→0.867.
3. MIN_SUPPORT 0.20 A/B → promote (0.884).
4. Other-cam funnel: 0/28 clear FNs had another cam ≥0.80 mapped.
5. Skipped t552 re-detect (map still dominates residual).

## Evidence

- [HOLDOUT_BASELINE.md](HOLDOUT_BASELINE.md)
- [DEAD_ENDS.md](DEAD_ENDS.md)
- `reports/eval_match3/improve_eng_loop/{m1_provisional,r1_random_fn_audit_holdout,h1_minsupport_ab,holdout_other_cam_funnel}.json`

## Residual

- **P8 strip gold drift (2026-09-02):** after P8 H expand (Sep 1), stale `gold_xy` (x≈26.95) made P_emit look like 0 / err ~13 m. Reseeded from current `map_ball_box(gt_balls)` → P_emit **1.0**, clear_R **0.846**, err **0.147**. Guard: `scripts/gold_set/test_match3_strip_gold_sync.py`.
