# Match 3 clear-ball (product-wide)

Secondary Phase 1 gate: **clear-ball R ≥ 0.80** on the product multi-cam fuse path, without trading **P_emit ≥ 0.80**. Emit stays **conf ≥ 0.80**; agree ≤ **4 m**; Pitch 1 meters. **MIN_SUPPORT = 0.20** (H1 promote after holdout A/B).

Canonical improve loop: [MATCH3_MULTICAM_IMPROVE_PLAN.md](MATCH3_MULTICAM_IMPROVE_PLAN.md). Defish: [MATCH3_DEFISH.md](MATCH3_DEFISH.md).

## Current numbers (product F0)

| Pack | clear_ball_proxy_R / clear_R | P_emit |
|------|-----------------------------:|-------:|
| **Holdout** (seed 20260821) | **0.884** | — |
| Tune random | 0.625 | — |
| Strip P10 | **0.879** | 1.0 |
| Strip P8 | **0.846** | **1.0** |

Holdout freeze was **0.556**; map-first (P10 hull + MIN_SUPPORT 0.20) lifted it above **0.80** with strip precision held. L1 white-line re-click on P8/P9 (RT ≤ 0.15 m, hulls kept) → P10 clear_R **0.876→0.879**; P8 / holdout unchanged; strip P held.

**P8 strip gold (2026-09-02):** Sep 1 P8 landmark/H expand left Aug-21 `gold_xy` stale (x pinned at north endline ≈26.95 → false P_emit 0 / err ~13 m). Reseeded pitch gold from current `map_ball_box(gt_balls)`; pixels unchanged. Guard: `scripts/gold_set/test_match3_strip_gold_sync.py`.

## What worked

1. Spot-check holdout P10 `low_support` FNs (in-FOV) — [`reports/ball_testing/holdout_map_fn_spot/`](../../reports/ball_testing/holdout_map_fn_spot/).
2. Add P10 `hull_image_points` (lower FOV; H unchanged).
3. Promote `MIN_SUPPORT` **0.20** (`h1_minsupport_ab.json`).
4. Other-cam funnel: no silent fuse with other cam ≥0.80 on residual FNs.
5. L1 honest re-click P8/P9 posts/box (≤18 px on bright line pixels); `test_match3_l1.py` pass; no clip-only hull.

Skipped one-clip det on t552 (map FNs still majority).

## Anti-overfit

- Gate = **holdout**, not tune five.
- Do not hull clip-only cams (e.g. P_Goal1 on one seed).
- Do not lower emit / widen agree / stronger k1 alone.

## Notes / dead ends

- [`reports/ball_testing/HOLDOUT_BASELINE.md`](../../reports/ball_testing/HOLDOUT_BASELINE.md)
- [`reports/ball_testing/DEAD_ENDS.md`](../../reports/ball_testing/DEAD_ENDS.md)
- [`reports/ball_testing/CLEAR_BALL_FRONT.md`](../../reports/ball_testing/CLEAR_BALL_FRONT.md)
