# Team labeling (A/B) — eng-loop prompt

**Status:** implemented → gate all 20 ≥ 9/10 (`PASS` on frame 2400).

## Goal

Label which **team** each mapped player is on for Match 3 coach review:

- Pitch 1 dots: **Team 0 / Team 1** colors (not all gray)
- Live RF-DETR path (boxes ON), including **Best camera (ball)** after quad fill
- Precision-first: unsure → **unassigned / gray**, never invent a flip

Phase 1 scope: color clustering (`docs/product/PHASE1_SCOPE.md`). Not learned re-ID.

## Current gap

`fuse_live_dets_for_pitch` sets `team: -1` for every live player. Export CSV may have `Team_ID`; live mosaic → pitch does not. `TeamClusterer` exists (`src/perception/team_id.py`) but still assumes FIFA **105×68** for GK/spatial heuristics — use **Pitch 1** meters (`docs/product/PITCH1_DIMENSIONS.md`).

## Gate

```bash
python3 scripts/eng_loop_team_label.py
```

All **20** scores ≥ **9.0**. Report:

`reports/eval_match3/improve_eng_loop/team_label/scores.json`

Artifacts: `pitch_teams_f{FRAME}.jpg`, `crop_strip.jpg` (optional).

## Top 20 components

| # | Component | Done when (≥9) |
|---|-----------|----------------|
| 01 | **Import / wire** | Review can call a single `assign_teams_live(...)` without ImportError / stale reload |
| 02 | **Torso crop** | Jersey crop = upper ~55% of player box (not full legs / grass strip) |
| 03 | **Green suppress** | Pitch green HSV masked out before dominant jersey hue |
| 04 | **Feature vector** | Stable HSV (or Lab) feature per crop; NaN/empty → skip |
| 05 | **Min samples** | Frame/window fit only if ≥ **6** valid player crops (else all unassigned) |
| 06 | **K=2 fit** | Two centroids; silhouette / centroid distance shows kits separable when both present |
| 07 | **Assign 0/1** | Mapped live players get `team_id` ∈ {0,1} when conf high |
| 08 | **Unsure → gray** | Conf below gate or outlier → `team_id=-1` (precision-first) |
| 09 | **Pitch 1 spatial** | Any GK/box heuristic uses Pitch 1 length/width — **not** 105×68 |
| 10 | **Ref / non-kit** | Clear non-team kits (black/yellow ref-like) not forced into A/B if outlier |
| 11 | **Multi-cam vote** | Same fused person: majority team across cams; ties → max-conf cam |
| 12 | **Label lock** | Team 0/1 meaning stable within a session (no random swap each frame); lock by hue order (e.g. warmer centroid = 0) |
| 13 | **Both teams when visible** | On a frame with two kits, pitch shows **n0≥1 and n1≥1** |
| 14 | **Pitch panel colors** | T0 / T1 match `pitch1_panel` legend (blue/red as coded); gray = unassigned |
| 15 | **Live fuse integration** | `fuse_live_dets_for_pitch` (or post-pass) fills team from crops on detect frames |
| 16 | **Best-ball compatible** | After `fill_quad_dets_for_pitch`, teams still assigned across quads |
| 17 | **No ball as player** | Ball dets never get a team |
| 18 | **Count sanity** | `#assigned + #gray == #fused players`; assigned ≤ fused |
| 19 | **Unit tests** | Synthetic two-color crops → correct 0/1 split; green-only crop → unassigned/outlier |
| 20 | **Product ready** | Mean of hard gates (07,08,11,13,15,16,18) ≥ 9; eng-loop **PASS** |

## Hard no

- FIFA 105×68 / 18-yard / penalty-spot assumptions for Match 3
- Force every detection into team 0 or 1
- Swap team colors randomly frame-to-frame
- Phase 2 re-ID / jersey number OCR as a Phase 1 requirement
- Lower ball emit gate or change mosaic layout / defish defaults

## Wire (implementation target)

| Piece | Where |
|-------|--------|
| Feature + cluster | `src/perception/team_id.py` (Pitch 1 dims) and/or thin `src/review/team_live.py` |
| Crop helper | torso crop + green suppress |
| Live map | `src/review/multicam_fuse.py` — set `team` on player_pts before fuse vote |
| Need frames | detect frame BGR per cam (from mosaic ensure path) — pass crops or `(cam → frame)` into fuse |
| Panel | already colors by team (`pitch1_panel.py`) |
| Eng-loop | `scripts/eng_loop_team_label.py` |
| Unit | `scripts/test_team_live_label.py` |

## Scoring frames

Primary: Match 3 quads @ **frame 2400** (defish ON). Smoke: 1200, 3600.

## Done definition

Coach sees **blue vs red** (or legend T0/T1) dots on Pitch 1 for clear kits; gray only when unsure; Best-ball still shows full multi-cam roster with teams; eng-loop PASS.

## Shipped

| Piece | Path |
|-------|------|
| Live label | `src/review/team_live.py` (mid-torso + blue/white/yellow kit fractions) |
| Fuse + `{cam}__bgr` | `src/review/multicam_fuse.py`, `cam_mosaic.py` |
| Eng-loop | `scripts/eng_loop_team_label.py` (includes crop BGR separation ≥28) |
| Unit | `scripts/test_team_live_label.py` |
| Scores | `reports/eval_match3/improve_eng_loop/team_label/scores.json` |

**v2 fix:** Lab features collapsed on Match 3 kits → switched to kit-fraction features; tighter torso band; grass/leg crops rejected; vision `bgr_sep` gate.
