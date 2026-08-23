# PROMPT evaluation — fuse player ID stability

**Score: 8.7/10 — APPROVED**

| Criterion | /10 |
|-----------|-----|
| Root cause (j+1 slot ids) | 9 |
| Bounded fuse-layer change | 9 |
| Event regression coverage | 9 |
| Risk to team_stable / ghost holds | 8 |
| Coach-visible attribution | 8 |

**Baseline:** ~13 id swaps on small xy steps in 10.5–11.5 s carry window; dribble attributes player 5 vs 6.

**Post-run (2026-08-23):** `eng_loop_fuse_player_id_stable.py` **PASS** — swaps **13→0**, unique carrier id **7→1**; event regression green on raw timeline.
