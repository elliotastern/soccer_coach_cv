# Match 3 clear-ball (product-wide)

Secondary Phase 1 gate: **clear-ball R ≥ 0.80** on the product multi-cam fuse path, without trading **P_emit ≥ 0.80**. Emit stays **conf ≥ 0.80**; agree ≤ **4 m**; Pitch 1 meters. **MIN_SUPPORT = 0.20** (H1 promote after holdout A/B).

Canonical improve loop: [MATCH3_MULTICAM_IMPROVE_PLAN.md](MATCH3_MULTICAM_IMPROVE_PLAN.md). Defish: [MATCH3_DEFISH.md](MATCH3_DEFISH.md).

## Current numbers (product F0)

| Pack | clear_ball_proxy_R / clear_R | P_emit |
|------|-----------------------------:|-------:|
| **Holdout** (seed 20260821) | **0.884** | — |
| Tune random | 0.625 | — |
| Strip P10 | 0.876 | 1.0 |
| Strip P8 | 0.801 | 0.966 |

Holdout freeze was **0.556**; map-first (P10 hull + MIN_SUPPORT 0.20) lifted it above **0.80** with strip precision held.

## What worked

1. Spot-check holdout P10 `low_support` FNs (in-FOV) — [`reports/ball_testing/holdout_map_fn_spot/`](../../reports/ball_testing/holdout_map_fn_spot/).
2. Add P10 `hull_image_points` (lower FOV; H unchanged).
3. Promote `MIN_SUPPORT` **0.20** (`h1_minsupport_ab.json`).
4. Other-cam funnel: no silent fuse with other cam ≥0.80 on residual FNs.

Skipped one-clip det on t552 (map FNs still majority).

## Anti-overfit

- Gate = **holdout**, not tune five.
- Do not hull clip-only cams (e.g. P_Goal1 on one seed).
- Do not lower emit / widen agree / stronger k1 alone.

## Notes / dead ends

- [`reports/ball_testing/HOLDOUT_BASELINE.md`](../../reports/ball_testing/HOLDOUT_BASELINE.md)
- [`reports/ball_testing/DEAD_ENDS.md`](../../reports/ball_testing/DEAD_ENDS.md)
- [`reports/ball_testing/CLEAR_BALL_FRONT.md`](../../reports/ball_testing/CLEAR_BALL_FRONT.md)
