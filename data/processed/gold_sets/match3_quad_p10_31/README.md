# Match 3 M1 strip — quad P10 @ 0:31

Synced 5 s clip (`quad_P10_t00031.0s`) with provisional clear-ball GT on **P10**.

- `labels.json` — per-frame P10 `gt_balls` + `gold_xy` (Pitch 1 m)
- `review/frames/` — local JPGs (not for git)
- Score: `python3 scripts/gold_set/score_match3_ball_m1.py`

Seed is detector-based. Correct boxes before claiming final P_emit.
