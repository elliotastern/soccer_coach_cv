# Overnight fusion — continuing (v14 era)

## Top 5 (ranked for goal: fusion → ball / player / kit)

| Rank | Lever | Overfit | Why |
|-----:|-------|---------|-----|
| **1** | **Re-measure C2 quad funnel under product v14; lift P9 t655 if still &lt;0.80** | **Low** (held strips; no holdout train; no fuse retune) | Eng-loop `c2_det=9` still cites stale v10/v12 funnel. Product is v14 — scoreboard lag is the first honest gap. |
| 2 | Human re-click P1/P6 existing landmarks | Medium | Only path for `l2_overlap`; skipped as overnight Top-1 (overfit + needs human) |
| 3 | Match-4 add P1/P6/goals for kit multcam_frac | Low | Kit consensus capped ~8.6 at mfc=0.22 even with perfect agree; Catch currently down |
| 4 | Kit agree-within-cluster (not merge_m) | Low–medium | Second term only; cannot alone hit consensus 9 |
| 5 | v15 residual on ~11 soft holdout FNs | Low | Diminishing (holdout already 0.960; funnel 0 other-cam≥0.80) |

**Top-1 does not overfit** under re-measure + optional specialty on P9 t655 only (holdout gallery out of train). Proceed with 10-subgoal loop.

## Already done (do not redo)

- v14 ball promote · kit merge widen dead · auto-H/snap dead · product_goals **10**
