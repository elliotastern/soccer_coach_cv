# Heuristic events — eng-loop prompt

## Goal

Ship Phase 1 **pass / shot / recovery** heuristics with **P_emit ≥ 0.80** on Pitch 1 meters.

## Gate

```bash
python3 scripts/eng_loop_heuristic_events.py
```

All **20** scores ≥ **9.0**. Report:

`reports/eval_match3/improve_eng_loop/heuristic_events/scores.json`

## Top 20 components

| # | Component | Done when (≥9) |
|---|-----------|----------------|
| 01 | Prompt / wire | This PROMPT exists |
| 02 | Pitch 1 half-length | 26.95 m (not FIFA) |
| 03 | No FIFA 52.5 | Absent from `events.py` |
| 04 | Emit conf ≥ 0.80 | Constant + detector |
| 05 | Priority exclusive | ≤1 emit; shot wins near goal |
| 06 | Gold manifest | ≥4 scored synth clips |
| 07 | Offline runner | `emits.json` written |
| 08 | **P_emit synth** | ≥ 0.80 on synth gold |
| 08b | **P_emit real** | ≥ 0.80 on check25 product-fuse timeline |
| 09 | Weak pass clean | No FP on weak_none |
| 10 | Shot Goal2 | TP on synth_shot_goal2 |
| 11 | Strong pass | TP on synth_pass_strong |
| 12 | Recovery | TP on synth_recovery |
| 13 | Midfield ≠ shot | Fast midfield emits pass |
| 14 | Recall secondary | ≥ 0.5 on labeled |
| 15 | Dead ends doc | `DEAD_ENDS.md` |
| 16 | Front doc | `EVENTS_FRONT.md` |
| 17 | Improve plan | `HEURISTIC_EVENTS_IMPROVE_PLAN.md` |
| 18 | Unit E0 | `test_heuristic_events_e0.py` |
| 19 | Config emit | `emit_conf` in default.yaml |
| 20 | Product ready | Mean of core ≥ 9 |

## Hard no

- Lower emit conf
- FIFA meters
- Learned events
- Ball map retune for events

## Done

`eng_loop_heuristic_events.py` PASS; PHASE1_STATUS events row ≥7.
