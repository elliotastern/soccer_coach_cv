# PROMPT evaluation — fuse teleport mask

**Score: 8.8/10 — APPROVED**

| Criterion | /10 |
|-----------|-----|
| Targets fuse map glitch root cause | 9 |
| Complements existing ball-speed gate | 9 |
| Measurable audit + emit-on-teleport gate | 9 |
| Risk to valid dribble after stable segment | 8 |
| Does not fix upstream fuse xy | 8 |

**Baseline:** ball-speed mask only; 72 ball steps >3 m on fuse 15 s; player pid-5/10 jumps in carry window before stable segment at ~10.0 s.

**Run after approval (2026-08-23):** `eng_loop_fuse_teleport_mask.py` **PASS** — fuse **2 pass + 1 dribble**; dribble TP **1**; ball teleports audited **72**; **0** emits on teleport timestamps; `fuse_shot_recovery` regression green.
