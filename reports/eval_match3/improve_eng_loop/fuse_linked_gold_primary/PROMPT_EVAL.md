# PROMPT evaluation — fuse linked gold primary

**Score: 8.9/10 — APPROVED**

| Criterion | /10 |
|-----------|-----|
| Fixes carrier pid mismatch (5 vs 34) | 9 |
| Minimal wiring (build script + labels field) | 9 |
| Measurable carrier + swap gates | 9 |
| Does not replace upstream fuse xy quality | 8 |
| Coach handover still needs real `frames` QA | 8 |

**Baseline:** raw timeline dribble attributes slot id **5**; linked relink gives stable carrier **34** with 0 swaps in carry window.

**Run after approval (2026-08-23):** `eng_loop_fuse_linked_gold_primary.py` **PASS** — linked **2 pass + 1 dribble**; carrier **34**; carry swaps **0**; regression green.
