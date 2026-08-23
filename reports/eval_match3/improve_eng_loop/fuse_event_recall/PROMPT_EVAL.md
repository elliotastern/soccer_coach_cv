# PROMPT evaluation — fuse event recall

**Prompt:** `fuse_event_recall/PROMPT.md`  
**Evaluated before implementation:** 2026-08-23

## Rubric (each /10, need ≥8 average)

| Criterion | Score | Notes |
|-----------|------:|-------|
| **Problem clarity** | 9 | Separates batch spam (fixed) vs fuse recall gap |
| **Measurable gates** | 9 | TP, count band 1–4, P_emit, regression hooks |
| **Regression safety** | 9 | Parent loops + batch cap + tune bounds |
| **Wire map / commands** | 8 | Scripts named; timeline builder specified |
| **No forbidden scope** | 10 | No map retune, no emit drop, no single-frame revert |
| **Coach outcome** | 8 | “Visible carry → dribble flash” testable on 15 s mp4 |

**Average: 8.8/10** — **APPROVED to implement and run.**

## Risks logged

1. **Movement vs dribble** — priority puts dribble before movement; tune may still need min_carry drop on fuse xy.
2. **Timeline build cost** — ~225×4-cam detect; one-time gold build acceptable.
3. **Batch csv gaps** — 0 batch dribbles expected; gate is cap not minimum.

## Pre-run baseline (post-D3)

- Fuse 15 s: 0 dribble, 2 pass, 1 movement (`check_15s_s4/meta.json`)
- Batch replay: 0 dribble (`dribble_precision/scores.json`)
- Heuristic + dribble_precision: all gates 10/10

## Post-run (2026-08-23)

- Fuse timeline built; **3 emits** (2 pass + 1 movement), **P_emit 1.0**, carry TP on 11 s window.
- Dribble still **0** on fuse (player cling >2 m / ID swaps) — **movement** is correct carry emit.
- `eng_loop_fuse_event_recall.py` **PASS** all gates ≥9/10.
