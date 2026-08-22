# Events testing — dead ends

Log failed A/Bs here. Numbers live in JSON; this file is the anti-rerun list.

| Tried | Result | Why not again |
|-------|--------|---------------|
| FIFA `goal_x = 52.5` for shots | Missed all Pitch 1 goal-band plays (half ≈ 26.95) | Pitch 1 only |
| Dribble conf = 0.7 | Below Phase 1 emit gate | Raise quality bar or suppress (E2) |
| Multi-emit (pass+shot+…) same frame | Inflates FP / confuses export | Priority: shot > pass > recovery |
| Lower emit conf to “get more events” | Violates Phase 1 PoC | Forbidden |
| Score only synth gold | P_emit 1.0 but real check25 was **0.03** (36 emits, teleports) | Product gate = real fuse timeline |
| Raw frame-to-frame events on stride-15 fuse | Ball jumps 40–50 m → fake 150+ m/s passes/shots | `MAX_BALL_SPEED_M_S=40` + post-teleport settle |
| Recovery on goal-line possession cling | FP spam | Require close-from >2.5 m + move ≥0.8 m |
| Label goal-band leave as “shot” | Physics is clearance/pass (abs x decreases) | Shot only if moving toward goal line |
| Shared `EventDetector` across gold clips | Cooldown leaked → synth shot/recovery FN | Fresh detector per clip |

## Worked

| Change | Evidence |
|--------|----------|
| Pitch 1 half-length + emit gate | `test_heuristic_events_e0.py` |
| Teleport reject + cooldown + in-pitch | check25 `score_real.json` |
| Real pack labels from continuous xy | 3 pass windows; **P_emit_real 1.0** |
| Eng-loop kill switch on real | `08b_real_p_emit` in `scores.json` |
| Default thresholds | `t1_threshold_ab.json` |

## Residual

- Shot/recovery rare on this 25 s window (mostly clearances labeled pass).
- Stride-15 timeline still loses some sub-second events; denser timeline is compute-heavy.
- E2: dribble + movement still off.
