# Mosaic holdout scores (report-only)

Scored stride-15 **5-min fuse mosaic** emits against provisional holdout labels.

| Clip | mosaic emits in window | P_emit | note |
|------|----------------------:|-------:|------|
| `real_fuse_eval_20s` | dribble, pass | 0.0 | gold expects movement — mosaic stride-15 differs |
| `real_fuse_eval_49s` | move, dribble, move, pass, pass | 0.2 | stride-4 gold windows ≠ stride-15 times |
| `real_fuse_eval_69s` | pass, movement, dribble | **1.0** | provisional labels matched prior mosaic |

**Product primary gate** stays `real_fuse_15s` stride 4 (P_emit 1.0).

**Stride-4 @49s** (timeline score): P_emit **0.67** (2 tp movement+dribble, 1 fp late movement ~6.8s) — coach confirm late movement.

Artifacts: [MOSAIC_HOLDOUT_SCORE.json](MOSAIC_HOLDOUT_SCORE.json) · rate sheet [COACH_RATE_SHEET_mosaic_5min_fuse.md](COACH_RATE_SHEET_mosaic_5min_fuse.md)
