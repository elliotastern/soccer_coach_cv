# Top-1f: constrained landmark nudge (existing IDs only)

## Top 5 (now)

| Rank | Lever | Overfit | Status |
|-----:|-------|---------|--------|
| ~~1–e~~ | C2 under v14 | Low | Done (`c2_det=10`) |
| **1** | **Constrained nudge of existing P1/P6 clicks (≤N px) + kill-switches** | **Low if gated** | Proceed |
| 2 | Human free re-click | Medium | Fallback if nudge fails |
| 3 | Match-4 extra cams for kit | Low | Catch down |
| 4 | Kit agree-within-cluster | Low | Caps &lt;9 at mfc=0.22 |
| 5 | v15 soft-FN residual | Low | Diminishing |

## Why Top-1 is not overfit

Only moves **existing** landmark image points toward pitch-consistent projections within a pixel budget; promote only if strip `P_emit≥0.80` and holdout proxy `≥0.884`. No new IDs, no emit/agree change.

## 10 subgoals

1. Per-point residual baseline P1/P6
2. Implement constrained nudge (no write)
3. Sweep max_px ∈ {6,10,14,18}
4. Why: which max_px can hit RT≤0.15 without wild moves
5. Dry-run kill-switch plan
6. Apply best to trial calibs (temp)
7. Strip P_emit / clear_R vs v14 baseline
8. Holdout proxy vs gate 0.884
9. eng_loop `l2_overlap` ≥9
10. Promote or DEAD_ENDS + restore
