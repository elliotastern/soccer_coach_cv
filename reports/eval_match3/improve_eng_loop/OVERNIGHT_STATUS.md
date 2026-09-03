# Overnight Top-1 status board — COMPLETE

**Goal:** camera fusion 9/10+ via better ball/player/kit metrics  
**Top-1 (non-overfit):** detector soft-conf specialty — **DONE / PROMOTED v14**

## Results

| Metric | v13 | v14 |
|--------|----:|----:|
| Strip P10 clear_R | 0.888 | **0.929** |
| Strip P8 clear_R | 0.904 | **0.908** |
| Holdout proxy R | 0.893 | **0.960** |
| Holdout soft FNs | 26 | **11** |
| Other-cam≥0.80 funnel | 0/32 | **0/11** |

Product `ball_checkpoint` → `models/v14_residual_snaps/post_train/checkpoint.pth`

## Top 5 (next after Top-1)

1. ~~Detector soft-conf specialty~~ **done**
2. H/map on existing landmarks (L2 eng-loop still 6 — FOV locked; skip invent)
3. Player pitch-map stitch ← next for kit metrics
4. Kit-ref sticky
5. Opt-in 3D when agree

## Subgoals 1–10

All ≥9/10 for Top-1 loop (see overnight_subgoal_*_result.json / promote.json).
