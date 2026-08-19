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

**Scoring caveat:** gallery det caches were built with `--stride 2` (detect every other frame). Raw clear_R on all 60 fps labels under-counts because odd frames were never inferred. Primary strip metrics use **detect ticks** (even frames); `carry_neighbor_tick` approximates holding the last tick for product-like 30 Hz detect / 60 Hz emit.

Latest score (`score_match3_ball_m1.py`): **P_emit = 1.0**, **clear_ball_R ≈ 0.89** (detect ticks); raw ≈ 0.44; carry ≈ 0.95. Both PoC gates pass on primary.

### Fuse post A/B (F0–F2)

Run: `python3 scripts/gold_set/ab_match3_fuse_post.py` → `reports/eval_match3/improve_eng_loop/f_post_ab.json`

| Variant | P_emit | clear_ball_R | Notes |
|---------|--------|--------------|-------|
| baseline / F1 / F2 / F1+F2 | 1.0 | 0.889 | detect ticks; same on this strip |
| **F1+F2+F0** (winner) | 1.0 | **0.910** | sequential hold across silent frames |

F1/F2 are correctness (soft-dual fallthrough + max-conf solo); F0 hold is the measured R lift. Emit stays 0.80; agree stays 4 m.

**Gallery (human check):** 5 random video|pitch clips re-rendered with F0–F2 — http://127.0.0.1:8080/reports/eval_match3/pitchmap_gallery/ (`?v=fpost`). Post rated **8.5/10** after review (videos look solid; remaining gap is model conf, not fuse).

Gold pitch `(x, y)` = P10 bbox foot mapped through `P10_manual.json` (Pitch 1 meters).

## Status

Human-reviewed through frame ~194 on this pack (`human_reviewed`). Review UI (`/match3-m1`): drag to draw/replace box, Clear box, Save (rematches `gold_xy` via `/save_match3_m1_labels`).

Frames stay local (`review/frames/`); git tracks labels + manifest + review HTML.
