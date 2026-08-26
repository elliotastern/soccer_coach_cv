# Team Match 4 balance — eng-loop prompt

**Status:** Loop C — gates on mosaic `meta.json` after kit_mode=auto + fuse dedup. **90s PASS** (2026-08-25): collapse 58%→7%, both3 15%→76%.

## Goal

Match 4 white kits were collapsing to blue (58% collapse frames). After Loop A+B:

- Mean team-0 share in **0.35–0.65**
- Collapse frames (n1≤1 & n0≥5) **< 15%**
- Both kits ≥3 on **> 50%** of frames
- Mean fused players **≤ 14.5** (cap + merge)

## Gate

```bash
python3 scripts/eng_loop_team_match4_balance.py
# optional: --meta path/to/meta.json
```

Report: `reports/eval_match3/improve_eng_loop/team_match4_balance/scores.json`

Baseline snapshot: `team_match4_baseline.json` (auto-created on first run).

## Rerender (Loop D)

After code change, rebuild Match 4 mosaic meta:

```bash
# Catch preferred — 90s smoke:
python3 scripts/gold_set/render_phase1_check_mosaic.py \
  --start 0 --match-sec 90 --stride 15 --out-dir reports/eval_match3/improve_eng_loop/match4_5min \
  --out-file coach_mosaic_first_90s.mp4 --debug-cam
```

Re-run eng-loop; hard gates (03–06, 20) must be ≥ 9.0.

## Related

- Loop A: `team_core.py` / `team_live.py` — `kit_mode: auto`
- Loop B: `multicam_fuse.py` — cap 14, merge 2.2m, solo team conf 0.55
- Match 3 regression: `eng_loop_team_label.py` uses `kit_mode=match3`
