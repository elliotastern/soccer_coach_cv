# Team ID top-10 strategy grid

Compare 10 team-ID strategies on Match 4 first 90s (stride 15) with plain vs SAHI detection.

## Run

```bash
python3 scripts/eval_team_id_strategy_grid.py \
  --start 0 --match-sec 90 --stride 15 \
  --sahi plain,sahi --strategies all
```

Catch (recommended): `bash scripts/run_team_id_grid_on_catch.sh`

Apply winner: `python3 scripts/apply_team_id_grid_winner.py`

## Latest run (2026-08-26, Catch Match 4)

- **Production strategy:** S11/S13 `auto_traj_no_gray` (traj vote + soft pixel + no gray)
- **90s verify:** `coach_mosaic_s13_traj_pixel_90s.mp4` — fr750 **8/6/0**, mean blue **0.61**, **0% gray**
- **5min verify:** `match4_5min_s13/coach_mosaic_pitch_5min.mp4` — eng-loop **PASS** (mean blue 0.59, collapse 0.7%)
- Config: `configs/default.yaml` → `strategy: auto_traj_no_gray`
- Artifacts: `reports/eval_match3/team_id_strategy_grid_m4/`

## Outputs

- `ranking.json` — all runs scored
- `ranking.md` — sortable table + frame 750 spot check
- `m4_90s_det_plain.json` / `m4_90s_det_sahi.json` — detection cache

## Scoring axes

| Axis | Meaning |
|------|---------|
| **Consensus** | Multi-cam cluster agreement |
| **Balanced** | Session blue share, collapse, both3 |
| **Flickering** | Sticky keep + share swing rate |
| **SAHI** | Column only (plain vs sahi det) |

Composite: `0.30×consensus + 0.35×balanced + 0.35×flickering`
