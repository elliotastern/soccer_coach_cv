# Phase 1 overnight scorecard (out of 10)

Saved after overnight fusion / ball / kit runs (2026-09-03).

| Pillar | Before overnight | After overnight | Δ |
|--------|----------------:|----------------:|---|
| Ball (strip clear_R / emit) | 8.5 (v13 ~0.89/0.90; P_emit ~1.0) | 9.3 (v14 0.93/0.91; P_emit ≥0.97) | +0.8 |
| Player map / fuse | 9.0 (already strong) | 9.5 (P1/P6 H fixed; l2_overlap 6→10) | +0.5 |
| Kit (Match-4 product / quad) | 7.1 consensus · 9.0 composite | 7.7 consensus · 9.2 composite | +0.6 cons |
| Kit path (full-cam proof) | ~6.5 (quad-only ceiling) | 9.2 (Match-3 full6 freeze 9.11/9.31) | +2.7 path |
| Camera fusion eng-loop | ~8.6 avg (l2=6, c2=9, goals=9) | 10 on Match-3 lanes · 7.7 Match-4 kit | Match-3 fixed; Match-4 exposed |
| Delivery readiness (Catch Match-4) | 6 (quad-only handover) | 6 (same — Catch still offline) | 0 |

## Overall Phase 1

| | Score |
|--|------:|
| Before | ~8.0 / 10 |
| After | ~8.7 / 10 |
| If Catch adds P1/P6 (proven path) | ~9.2 / 10 |

## Notes

- v14 ball weights are gitignored under `models/`; product config points at `models/v14_residual_snaps/post_train/checkpoint.pth`.
- Match-4 kit consensus ≥9 still needs Catch batch with P1/P6 + kit-ref — see [CATCH_MATCH4_KIT_CONSENSUS_9.md](CATCH_MATCH4_KIT_CONSENSUS_9.md).
