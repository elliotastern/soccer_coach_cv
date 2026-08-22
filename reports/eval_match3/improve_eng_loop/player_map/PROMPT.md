# Player map quality — eng-loop

Map-first loop: landmarks / H / hull support for player Pitch 1 XY.

## Gate

`python3 scripts/gold_set/eng_loop_player_map.py` → all scores ≥ **9/10**.

## Pack

`python3 scripts/gold_set/player_map_pack.py`

## Hard no

- Do not lower ball `MIN_SUPPORT=0.20`
- Do not promote L1 that breaks P8 ball map
- Do not fuse-soften as primary while H crushes Goal2

## Ledger

See `DEAD_ENDS.md` in this folder.
