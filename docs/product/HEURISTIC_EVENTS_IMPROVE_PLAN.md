# Heuristic events — improve loop plan

Engineering loop for Phase 1 **heuristic events** on Pitch 1 xy (Match 3 product map). Precision-first: **P_emit ≥ 0.80**. Below emit conf → do nothing.

## Goals

| Role | Metric | Target |
|------|--------|--------|
| **Primary** | Precision of **emitted** events (`P_emit`) | **≥ 0.80** |
| **Secondary** | Recall on labeled clear events | report-only |
| **Not the bar** | All-touch / every dribble frame | Phase 2 / E2 |

First slice: **pass**, **shot**, **recovery**. Dribble + movement = E2 later.

## Why current fails

Scaffold in `src/events/events.py` used FIFA `goal_x = 52.5`, dribble conf 0.7 (&lt; 0.80), multi-emit per frame, no gold/eng-loop. Status **2/10**.

## Loop steps

| Step | Do | Done when |
|------|----|-----------|
| **E0** | Pitch 1 half-length, emit gate, priority | Unit `test_heuristic_events_e0.py` |
| **G1** | Tiny gold `match3_events_v1` | Manifest + synth labels |
| **S1** | Offline timeline runner | `emits.json` per clip |
| **M1** | Eng-loop: **P_emit ≥ 0.80** on **real** check25 fuse (+ synth regression) | `scores.json` exit 0; `08b_real_p_emit` |
| **M1b** | Teleport reject, cooldown, toward-goal shot, recovery close-from | Real FP 35→0 |
| **T1** | Threshold A/B + DEAD_ENDS | Gate holds |
| **V1** | Overlay / status row | PHASE1_STATUS events ≥7 |
| **E2** | Dribble + movement | Same gate |

## Hard no

- Learned event models / Phase 2 RNN
- Emit conf &lt; 0.80
- FIFA 105×68 or `goal_x = 52.5`
- Retune ball map/hull for event recall

## Wire

| Piece | Path |
|-------|------|
| Detector | `src/events/events.py` |
| Config | `configs/default.yaml` → `events.*` |
| Gold | `data/processed/gold_sets/match3_events_v1/` |
| Offline | `scripts/gold_set/run_match3_events_offline.py` |
| Eng-loop | `scripts/eng_loop_heuristic_events.py` |
| Scores | `reports/eval_match3/improve_eng_loop/heuristic_events/scores.json` |
| Dead ends | `reports/events_testing/DEAD_ENDS.md` |
