# Events front (Phase 1 first slice)

**Target:** pass / shot / recovery · **P_emit ≥ 0.80** on **real** Match 3 product-fuse timeline · emit conf ≥ 0.80 · Pitch 1.

| Item | Status |
|------|--------|
| E0 Pitch1 + gate | Done |
| Teleport / cooldown / toward-goal shot | Done |
| G1 synth + check25 continuous labels | Done |
| Product timeline | `check25_human/timeline.json` (stride 15, defish fuse) |
| **Real P_emit** | **1.0** (recall 1.0) — `score_real.json` |
| Eng-loop | **PASS** (`08b_real_p_emit`) |
| Scores | `reports/eval_match3/improve_eng_loop/heuristic_events/scores.json` |
| E2 dribble/movement | Later |

Rebuild timeline:

```bash
python3 scripts/gold_set/build_check25_event_timeline.py --match-sec 25 --stride 15
python3 scripts/eng_loop_heuristic_events.py
```

Plan: `docs/product/HEURISTIC_EVENTS_IMPROVE_PLAN.md`.
