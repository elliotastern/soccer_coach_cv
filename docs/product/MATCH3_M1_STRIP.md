# Match 3 M1 gold strips

Human-confirm focus-cam ball boxes, then Save (rematches `gold_xy`).

| Pack | Focus | Clock | Review |
|------|-------|-------|--------|
| `match3_quad_p10_31` | **P10** | 0:31–0:36 | http://127.0.0.1:8080/match3-m1-p10 |
| `match3_quad_p8_87` | **P8** | 1:27–1:32 | http://127.0.0.1:8080/match3-m1-p8 |

**Hub:** `python3 serve_viewer.py` → http://127.0.0.1:8080/match3-m1  
**Rebuild review UI:** `python3 scripts/gold_set/build_match3_m1_review.py`  
**Score:** `python3 scripts/gold_set/score_match3_ball_m1.py`

## Metrics

| Metric | Definition | Target |
|--------|------------|--------|
| **P_emit** | Among fuse emits on frames with `gold_xy`, fraction with pitch error ≤ **4 m** | ≥ 0.80 (stretch ≥ 0.90) |
| **clear_ball_R** | Emit rate on frames marked clear on the focus cam (side ≥ 25 px) | ≥ 0.80 |

**Scoring caveat:** gallery det caches were built with `--stride 2` (detect every other frame). Raw clear_R on all 60 fps labels under-counts because odd frames were never inferred. Primary strip metrics use **detect ticks** (even frames); `carry_neighbor_tick` / F0 hold approximates product 30 Hz detect / 60 Hz emit.

## How to label

1. Open hub → pick **P10** first (already partly reviewed), then **P8**.
2. Prefer **Next clear** to jump seeded clear frames.
3. Drag to correct the ball box; **Clear box** if no ball / not clear.
4. **Save** (or ⌘/Ctrl+S) — rematches pitch `gold_xy` via focus-cam H + current hull/defish.
5. When done, `human_reviewed` is set on labels + manifest (`provisional=false`).

## Build

```bash
# P10 strip (original builder)
python3 scripts/gold_set/build_match3_m1_strip.py
# P8 strip
python3 scripts/gold_set/build_match3_strip.py --stem quad_P8_t00087.0s --focus P8 --pack match3_quad_p8_87 --clock "1:27-1:32"
python3 scripts/gold_set/build_match3_m1_review.py
```

Frames stay local (`review/frames/`); git tracks labels + manifest + review HTML.

## Status

Both packs start **provisional** until you Save from the review UI. After confirm, re-run `score_match3_ball_m1.py` + `eng_loop_match3_improve.py`.
