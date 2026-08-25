# Events front (Phase 1 first slice)

**Target:** pass / shot / recovery · **P_emit ≥ 0.80** on **real** Match 3 product-fuse timeline · emit conf ≥ 0.80 · Pitch 1.

| Item | Status |
|------|--------|
| E0 Pitch1 + gate | Done |
| Teleport / cooldown / toward-goal shot | Done |
| G1 synth + check25 continuous labels | Done |
| Product timeline (handover) | `real_fuse_15s` stride **4** (`check_15s_s4`) |
| **Fuse 15s P_emit** | **1.0** — 2 pass + 1 dribble |
| Holdout pass window | `real_fuse_holdout_pass` — outside handover 15s |
| Eng-loop | **PASS** — primary gate = fuse 15s; check25 stride 15 report-only |

Rebuild timeline:

```bash
python3 scripts/gold_set/build_fuse_15s_timeline.py
python3 scripts/gold_set/build_fuse_holdout_timeline.py
python3 scripts/eng_loop_heuristic_events.py
```

Plan: `docs/product/HEURISTIC_EVENTS_IMPROVE_PLAN.md`.
