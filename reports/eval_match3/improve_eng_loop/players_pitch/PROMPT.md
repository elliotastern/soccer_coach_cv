# Players on video + Pitch 1 — eng-loop

Goal: coach mosaic shows RF-DETR boxes; Pitch 1 panel shows fused mapped people with **player recall** (softer than ball) without ghost flood.

## Gate

`python3 scripts/eng_loop_players_pitch.py` → all **20** scores ≥ **9/10**.

Report: `reports/eval_match3/improve_eng_loop/players_pitch/scores.json`  
Funnel: `python3 scripts/gold_set/player_map_funnel.py` → `d1_player_map_funnel.json`

## What maps boxes → dots

1. `player_det_ok` — drop weak / tiny boxes (`PLAYER_MIN_CONF=0.50`). Mosaic still draws these.
2. `map_player_box` — foot → H with **`PLAYER_MIN_SUPPORT=0.10`** (ball stays **0.20**).
3. Live fuse: cluster merge 1.8 m (3.2 m goal box) + **live** solo/ghost floors (`0.40` / `0.35`).
4. Max-conf foot per cluster (not mean).

## Constants

| Name | Value | Where |
|------|------:|-------|
| PLAYER_MIN_CONF | 0.50 | multicam_fuse |
| PLAYER_MIN_SUPPORT | 0.10 | match3_xy (players only) |
| MIN_SUPPORT (ball) | 0.20 | match3_xy — do not lower |
| PLAYER_LIVE_SOLO_CONF | 0.40 | multicam_fuse live |
| PLAYER_LIVE_GHOST_CONF | 0.35 | multicam_fuse live |
| PLAYER_MERGE_M | 1.8 | multicam_fuse |

## Unit

`python3 scripts/test_player_pitch_fuse.py`

## Dead ends

`DEAD_ENDS.md` in this folder.
