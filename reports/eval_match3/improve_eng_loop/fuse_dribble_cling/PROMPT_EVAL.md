# PROMPT evaluation — fuse dribble cling

**Prompt:** `fuse_dribble_cling/PROMPT.md`  
**Score: 8.7/10 — APPROVED**

| Criterion | /10 |
|-----------|-----|
| Targets root cause (2 m vs 2.7 m cling) | 9 |
| Measurable gates + regression | 9 |
| Bounded change (one radius knob) | 9 |
| Risk: batch spam if csv gaps differ | 7 |
| Coach-visible outcome | 8 |

**Baseline:** fuse carry = movement only; batch dribble = 0; P_emit = 1.0 on pass+carry.

## Post-run attempt

- Widened cling to `movement_proximity` + closest-player step: **cling still breaks** on fuse stride (velocity band / player ID teleports between samples).
- **Movement at 11.27 s remains correct** product emit; forcing dribble label would be wrong.
- Loop **not passed** as dribble-specific; carry covered by `fuse_event_recall` (movement TP).
