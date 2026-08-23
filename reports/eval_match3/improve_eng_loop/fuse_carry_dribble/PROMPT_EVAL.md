# PROMPT evaluation — fuse carry dribble

**Pre-run score: 8.6/10 — APPROVED**

| Criterion | /10 |
|-----------|-----|
| Root cause (ID swap on fuse) | 9 |
| Bounded detector change | 9 |
| Regression gates | 9 |
| Risk to batch spam | 7 |
| Coach-visible test | 8 |

**Post-run (2026-08-23):** `eng_loop_fuse_carry_dribble.py` **PASS** — all components **10/10**. Fuse 15 s: 2 pass + **1 dribble** @ 11.33 s (P_emit 1.0, batch dribble 0). Parent loops green.

**Shipped extras:** stride-4 window (2 steps), movement deferral while partial carry &lt; min, cooldown bypass for in-progress carry, float epsilon on min_carry.
