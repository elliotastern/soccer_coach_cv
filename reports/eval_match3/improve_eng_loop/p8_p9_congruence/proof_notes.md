# P8 / P9 north-end proof notes

**Date:** 2026-08-22  
**Baseline:** `749eaae` (before changing cameras)

## P8 (`p8-005.mp4`)

Filename id **P8**; physical mount verified **north-right** (−y). In the north-goal still the camera sits on the **right** touchline side of Pitch 1 (P1’s right when looking north). Product had treated P8 as north-left; diagram chip, mosaic cell, and landmark touchline names were aligned to **north-right** after touchline mirror (`near`↔`far` on saved clicks).

## P9 (`P9-004.mp4`)

Filename id **P9**; physical mount verified **north-left** (+y). Mirror of P8 on the opposite touchline. Mosaic top-right cell and diagram chip moved to north-left; calib touchline mirror applied to saved clicks.

## Congruence gates

- `eng_loop_p8_p9_congruence.py` → **9.0/10** (P9 map sample thin at fr3050 but pass)
- Mosaic grid `P10|P9 / P7|P8`, diagram CAM_XY swapped, calibs refit

## Smoke clip

`reports/eval_match3/improve_eng_loop/player_map/check_15s_s4/coach_mosaic_pitch_11-12s.mp4` rerendered at frame 3050.
