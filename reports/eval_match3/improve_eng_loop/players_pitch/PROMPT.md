# Players on video + Pitch 1 — eng-loop

Goal: coach mosaic player boxes and Pitch 1 dots stay precision-first and consistent.

## Gate

`python3 scripts/eng_loop_players_pitch.py` → all **20** scores ≥ **9/10**.

Report: `reports/eval_match3/improve_eng_loop/players_pitch/scores.json`

## What fixed map inconsistency

1. `player_det_ok` — drop weak / tiny / board-like boxes (`PLAYER_MIN_CONF=0.50`).
2. Mosaic filter requires **pitch map** for players (same idea as ball).
3. Live fuse uses **max-conf foot** (not cluster mean) + weak-solo / ghost prune.

## Constants (`src/review/multicam_fuse.py`)

| Name | Value |
|------|------:|
| PLAYER_MIN_CONF | 0.50 |
| PLAYER_MIN_H | 40 |
| PLAYER_MIN_AREA | 800 |
| PLAYER_SOLO_CONF | 0.60 |
| PLAYER_GHOST_CONF | 0.55 |
| PLAYER_MERGE_M | 1.8 |

## Unit

`python3 scripts/test_player_pitch_fuse.py`
