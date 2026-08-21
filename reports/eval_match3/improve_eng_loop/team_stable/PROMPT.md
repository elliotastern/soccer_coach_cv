# Team label **stability** — eng-loop prompt

**Status:** implemented → cheap survey lift (adaptive green + hue hist + vote buffer + Pitch 1 goal-box prior). Eng-loop gate unchanged (≥9).

## Goal

Stop Pitch 1 team colors **flickering / swapping** while scrubbing Match 3 review.

Phase 1 needs Team A/B via color clustering **and** coach-usable stability: same kit stays the same color across nearby frames.

## Problem (measured)

Frames 2300→2500: team counts flip almost every step; 2400→2450 looks like a **T0↔T1 identity swap** (n1 2→6), not real roster change. Per-frame K=2 re-fit + weak lock causes this.

## Gate

```bash
python3 scripts/eng_loop_team_stable.py
```

All **20** scores ≥ **9.0**. Report:

`reports/eval_match3/improve_eng_loop/team_stable/scores.json`

## Top 20 components

| # | Component | Done when (≥9) |
|---|-----------|----------------|
| 01 | **Prompt / wire** | `assign_teams` API + session state import without stale reload |
| 02 | **Session centroids** | First good fit stored; later frames **reuse** (no fresh K=2 every frame) |
| 03 | **Hue/kit lock** | Team 0 = bluer, Team 1 = whiter/yellower — fixed for the session |
| 04 | **No identity swap** | Across scrub window, T0 does not become “the white kit” |
| 05 | **Hard kit rules** | Clear blue → 0, clear white/yellow → 1 (same as v2) |
| 06 | **Unsure → gray** | Ambiguous / dark / grass crop → −1 |
| 07 | **Update centroids slowly** | Optional EMA of centroids (α≤0.15); never flip order |
| 08 | **Track sticky team** | If ByteTrack/pid absent: sticky by pitch xy within 2.5 m across ±N frames |
| 09 | **Sticky vote** | Prior team kept unless new conf ≥ sticky gate **and** disagrees |
| 10 | **Multi-cam vote** | Same person: majority; tie → gray (unchanged) |
| 11 | **Best-ball OK** | After quad fill, teams still assigned + stable |
| 12 | **Share flicker gate** | On scored frames: ≤ **1** large swing in blue-share (|Δ n0/(n0+n1)| ≥ 0.40) |
| 13 | **No swap gate** | Correlation of (n0−n1) sign stable, or kit-feature of T0 stays blue-dominant |
| 14 | **Per-frame still both kits** | When both kits visible, n0≥1 and n1≥1 on ≥3/5 frames |
| 15 | **Gray not explode** | Mean gray fraction ≤ 0.45 across window |
| 16 | **Live fuse integration** | `fuse_live_dets_for_pitch` uses session locker |
| 17 | **Review session hook** | Streamlit stores locker in `st.session_state` (reset on match change) |
| 18 | **Unit: no swap** | Fit once; second batch with swapped sample order keeps same 0/1 meaning |
| 19 | **Unit: sticky** | Prior team 0 + weak opposite → stays 0 |
| 20 | **Product ready** | Mean of 04,09,12,13,14,16 ≥ 9; eng-loop **PASS** |

## Hard no

- Re-run unconstrained K=2 every frame
- FIFA 105×68
- Force every player into 0/1
- Phase 2 re-ID / jersey numbers

## Wire

| Piece | Where |
|-------|--------|
| Session locker | `src/review/team_live.py` — `TeamSession` |
| Fuse | `multicam_fuse.fuse_live_dets_for_pitch(..., team_session=)` |
| App | `app.py` — `st.session_state.team_session` |
| Eng-loop | `scripts/eng_loop_team_stable.py` |
| Unit | `scripts/test_team_session_stable.py` |

## Done

Scrubbing ~2s of Match 3: blue/red dots do not thrash; unsure stays gray; eng-loop PASS.
