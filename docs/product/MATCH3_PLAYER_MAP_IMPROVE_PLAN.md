# Match 3 player map quality (map-first)

Loop for mosaic boxes → Pitch 1 XY. **Map geometry only** — do not soften ball gates or fuse knobs for recall.

## Graph

```
raw box → player_det_ok → map_player_box (support / off_pitch)
        → OK feet → collapse check (image vs pitch spread)
        → live fuse (unchanged)
```

## Steps

| Step | Script / artifact | Done when |
|------|-------------------|-----------|
| D1 pack | `scripts/gold_set/player_map_pack.py` → `d1_pack_reasons.json` | cam×reason matrix |
| D2 collapse | same → `d2_collapse.json` | crush flags when `m_per_100px < 0.40` |
| D3 P8 bands | same → `d3_p8_quadrant.json` | bottom `off_pitch` on **H_player**; product `map_player_box` |
| L1 | P8 landmarks / H if off_pitch or crush systemic | RT ≤ 0.15 m **and** ball kill switch |
| L2 | `scripts/gold_set/refit_p8_h_player_bottom.py` + P8 lower-zone fallback in `map_player_box` | bottom mapped_frac ≥ 0.5 on D3 pack; ball on **H** unchanged |
| H_p | `hull_image_points` if pack majority `low_support` in FOV | low_support↓; H unchanged |
| G | `scripts/gold_set/eng_loop_player_map.py` | all scores ≥ 9/10 |
| E | short check clip under `player_map/check_*` | visual proof |

## Hard no

- Lower ball `MIN_SUPPORT` (0.20) / emit 0.80 / drop hull / hull-for-one-seed
- FIFA meters; cam-colored dots; fuse soften as primary while H crushes
- Promote L1 refit that breaks P8 ball map (lower-FOV feet)

## Evidence

`reports/eval_match3/improve_eng_loop/player_map/`
