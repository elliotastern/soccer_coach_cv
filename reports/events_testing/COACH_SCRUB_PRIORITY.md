# Coach scrub — priority pass (do this first)

**Labeling UI:** `bash scripts/run_coach_emit_label_dashboard.sh` → http://127.0.0.1:8502

Watch: `reports/eval_match3/improve_eng_loop/match4_5min/coach_mosaic_pitch_5min.mp4`
Full sheet: [COACH_RATE_SHEET_mosaic_5min_fuse.md](COACH_RATE_SHEET_mosaic_5min_fuse.md)

Prioritized **27** / 54 emits (t≤90s, shot/recovery, or near batch pass).

| # | type | t_end | conf | why | batch nearby | ok? |
|---|------|------:|-----:|-----|--------------|-----|
| 1 | shot | 1.50 | 1.0 | rare | — |  |
| 2 | movement | 6.50 | 1.0 | batch_conflict | pass@5.8, pass@6.9 |  |
| 3 | movement | 7.75 | 0.993 | batch_conflict | pass@6.9, pass@8.6 |  |
| 4 | recovery | 10.00 | 0.85 | rare | pass@8.6, pass@9.8 |  |
| 5 | movement | 14.00 | 0.91 | early | — |  |
| 6 | dribble | 22.50 | 1.0 | early | — |  |
| 7 | pass | 24.75 | 1.0 | early | — |  |
| 8 | dribble | 36.50 | 1.0 | early | — |  |
| 9 | pass | 41.00 | 0.802 | early | — |  |
| 10 | movement | 50.50 | 0.945 | batch_conflict | pass@49.0, pass@50.3, pass@51.4 |  |
| 11 | dribble | 50.75 | 1.0 | batch_conflict | pass@50.3, pass@51.4 |  |
| 12 | movement | 54.00 | 0.883 | batch_conflict | pass@54.1, pass@55.2 |  |
| 13 | pass | 57.75 | 1.0 | batch_conflict | pass@56.5 |  |
| 14 | pass | 59.50 | 1.0 | early | — |  |
| 15 | pass | 66.50 | 0.999 | early | — |  |
| 16 | pass | 68.25 | 0.96 | batch_conflict | pass@68.7 |  |
| 17 | pass | 74.25 | 0.91 | early | — |  |
| 18 | movement | 76.75 | 1.0 | early | — |  |
| 19 | dribble | 77.00 | 1.0 | early | — |  |
| 20 | dribble | 88.75 | 1.0 | early | — |  |
| 22 | recovery | 143.50 | 0.85 | rare | — |  |
| 23 | recovery | 147.75 | 0.85 | rare | — |  |
| 31 | recovery | 179.75 | 0.85 | rare | — |  |
| 48 | recovery | 246.25 | 0.85 | rare | — |  |
| 50 | recovery | 280.50 | 0.85 | rare | — |  |
| 51 | shot | 282.25 | 1.0 | rare | — |  |
| 53 | recovery | 289.50 | 0.85 | rare | — |  |

## Decision after this pass

1. If **movement** mostly FP → consider stricter movement gate (only with fuse gold).
2. If **shot/recovery** mostly FP → document in DEAD_ENDS; don't lower emit conf.
3. @49s late movement (~6.8s stride-4): keep vs drop after stride-4 MP4 arrives.
