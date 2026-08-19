# Match 3 M1 gold strip

**Pack:** `data/processed/gold_sets/match3_quad_p10_31/`  
**Clip:** `quad_P10_t00031.0s` (Match 3 clock **0:31–0:36**), focus cam **P10**  
**Build:** `python3 scripts/gold_set/build_match3_m1_strip.py`  
**Score:** `python3 scripts/gold_set/score_match3_ball_m1.py`  
**Review:** `python serve_viewer.py` → http://127.0.0.1:8080/match3-m1

## Metrics

| Metric | Definition | Target |
|--------|------------|--------|
| **P_emit** | Among fuse emits on frames with `gold_xy`, fraction with pitch error ≤ **4 m** | ≥ 0.80 |
| **clear_ball_R** | Emit rate on frames with clear P10 GT (side ≥ 25 px, conf ≥ 0.55 seed) | ≥ 0.80 |

Gold pitch `(x, y)` = P10 bbox foot mapped through `P10_manual.json` (Pitch 1 meters).

## Status

Seeded from detector clear P10 boxes — **provisional**. Human-correct `labels.json` / boxes in the review UI before treating PoC as final. Frames stay local (`review/frames/`); git tracks labels + manifest + review HTML.
