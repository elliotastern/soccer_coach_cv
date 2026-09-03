# Overnight fusion — Top-1b after v14 ball promote

**Goal:** camera fusion 9/10+ → better ball / player / kit metrics.

## Top 5 (re-ranked after v14)

| Rank | Lever | Overfit | Notes |
|-----:|-------|---------|-------|
| **1** | **Player multi-cam merge → kit consensus** (Match-4 holdout-gated `PLAYER_MERGE_M_LIVE`) | **Low if holdout-gated** | Map saturated; solos sit ~2.3 m apart (just outside 2.2 m live merge). Kit consensus **7.1** limited by `multcam_frac≈0.22`. |
| 2 | Kit-ref sticky refinements (after multcam rises) | Low–medium | Already composite 9.01; consensus is the soft spot |
| 3 | Residual ball @1288 when Catch is up | Low | Holdout already 0.960 — diminishing |
| 4 | Opt-in 3D ball when cams agree | Low | Gated; limited when agree sparse |
| 5 | Landmark RT re-click P1/P6 | **High / low payoff** | eng_loop `l2_overlap=6`; re-DLT no-op; **do not invent** marks |

**Hard nos:** invent landmarks, lower ball EMIT/MIN_SUPPORT, smoke-only merge retune without Match-4 hold, hull-for-one-seed.

## Top-1 decision

**Proceed.** Smoke-only merge widen would overfit; **Match-4 tune/hold split A/B will not** if we only promote on hold consensus↑ without ghost flood.

### Why (≥9.4)

Evidence `overnight_top1b_player_kit_why.json`:
- det_ok == map_ok (map not bottleneck)
- 18→14 is multi-cam merge (4 merges), not solo_conf drops
- nearest cross-cam solo gaps from **2.32 m** upward (live merge 2.2 m barely misses)
- kit consensus 7.1 ↔ multcam_frac 0.22

### Prompt (≥9.5)

```
Raise kit consensus via PLAYER_MERGE_M_LIVE A/B on Match-4 90s cache.
Split frames: first half tune, second half hold (or even/odd).
Sweep merge_m ∈ {2.2, 2.5, 2.8, 3.2} with kit-ref session.
Promote only if HOLD consensus ≥ baseline+0.3 and multcam_frac↑,
and mean_players / collapse gates do not regress.
Never change knobs using only Match3 fr2400 smoke.
Ball gates unchanged.
```

## 10 subgoals

1. Confirm map ≠ bottleneck (funnel) — **done**
2. Quantify merge gap (cross-cam solo distances) — **done**
3. Freeze kit baseline consensus/multcam on Match-4 90s
4. Define tune/hold split (no leakage)
5. Sweep merge_m on tune; pick candidate
6. Score candidate on hold only
7. Ghost/collapse/mean_players kill-switches
8. Ball product clear_R kill-switch (unchanged v14)
9. If promote: set `PLAYER_MERGE_M_LIVE`; eng_loop players_pitch ≥9
10. Re-score kit consensus ≥8.5 target (stretch 9.0) or document H-span ceiling

## Status

- Ball Top-1a (v14): **promoted**
- Player/kit Top-1b: why locked; A/B next
