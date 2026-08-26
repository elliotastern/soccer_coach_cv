# Batch events audit (2026-08-25)

Emit gate: **0.8** · Match4 quad emit conf: **70** · Match4 full emit: **32**
All dribble caps OK: **True**

## Per-pack

| Pack | emits | conf≥0.8 | dribble | pass/min (emit) |
|------|------:|--------:|--------:|----------------:|
| match_4_5min/P10-match4 | 16 | 16 | 0 | 12.41 |
| match_4_5min/P10-match4_cumulative | 16 | 16 | 0 | 12.41 |
| match_4_5min/P7-match4 | 1 | 1 | 0 | 60.0 |
| match_4_5min/P7-match4_cumulative | 1 | 1 | 0 | 60.0 |
| match_4_5min/P8-match4 | 18 | 18 | 0 | 8.67 |
| match_4_5min/P8-match4_cumulative | 18 | 18 | 0 | 8.67 |
| match_4_5min/P9-match4 | 0 | 0 | 0 | 0 |
| match_4_5min/P9-match4_cumulative | 0 | 0 | 0 | 0 |
| match_4_full/P10-match4 | 16 | 16 | 0 | 12.41 |
| match_4_full/P10-match4_cumulative | 16 | 16 | 0 | 12.41 |
| full_match_2min/P1-006 | 0 | 0 | 0 | 0 |
| full_match_2min/P10-002 | 330 | 28 | 283 | 36.27 |
| full_match_2min/P10-002_cumulative | 330 | 28 | 283 | 36.27 |

## Fuse gold clips

- **real_fuse_15s**: P_emit=1.0 tp=3 fp=0
- **real_fuse_eval_20s**: P_emit=1.0 tp=1 fp=0
- **real_fuse_eval_49s**: P_emit=0.0 tp=0 fp=3
- **real_fuse_holdout_pass**: P_emit=1.0 tp=1 fp=0
