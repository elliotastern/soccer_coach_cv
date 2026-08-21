# Ball testing — dead ends (do not re-run as “new”)

Living ledger. After any failed A/B, add one row. Numbers live under `reports/eval_match3/improve_eng_loop/`; this file is the human memory.

| Tried | Metric / pack | Result | Why not again |
|-------|---------------|--------|----------------|
| Stronger fisheye k1 alone | Random / strips | No product-wide R fix | Geometry already locked; see MATCH3_DEFISH |
| Defish (D0) alone | Random clear proxy | ~0.30 → ~0.63; still &lt; 0.80 | Necessary, not sufficient |
| Quad / P6 hull expands (C1/C3) for “product-wide” | Quad ↑ (~0.84); random stuck ~0.53 | Seed/FOV overfit when chasing tune pack | Prefer **holdout** gate + systemic FOV only |
| Proxy score without F0 hold | Random clear proxy | **0.525** undercount | Measurement bug — product already holds |
| R0 wire F0 into `score_cache` | Same packs | **0.625** | Measurement only — **no product change** |
| Holdout pack built (seed 20260821) | New times; freeze R **0.556** | Honest gate set | Do not retune on these starts blindly |
| Lower EMIT_CONF / widen AGREE_M / drop hull | — | Forbidden | Phase 1 precision-first Hard no |
| Phase-2 RNN / occlusion merge for Phase 1 clear-R | — | Out of scope | product_phase_scope |
| t552 one-clip det (map-first residual) | Holdout after P10 hull | **Skipped** — map FNs 22 vs conf 6 | Conf not majority; no MPS burn |

## Worked (keep)

| Tried | Result |
|-------|--------|
| Spot-check holdout P10 `low_support` | 73 in-FOV feet → proceed hull |
| P10 `hull_image_points` (lower FOV) | Holdout R **0.556 → 0.867**; strip P_emit held |
| H1 MIN_SUPPORT **0.25 → 0.20** | Holdout R **0.867 → 0.884**; strip P held → **promoted** |
| Other-cam ≥0.80 on clear FNs | **0/28** — residual is map/det, not fuse drop |

## Residual (not fixed)

- Clip-specific `focus_map_fail` (e.g. P_Goal1 low_support on t1767) — do **not** hull for one seed.
- Systemic `mapped_conf_below_emit` on **P6** (2 holdout caches, small n) — only if holdout regresses later.

See [CLEAR_BALL_FRONT.md](CLEAR_BALL_FRONT.md) · [HOLDOUT_BASELINE.md](HOLDOUT_BASELINE.md).
