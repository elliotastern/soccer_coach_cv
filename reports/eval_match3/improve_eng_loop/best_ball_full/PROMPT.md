# Best-ball view — full pitch detections

**Status:** implemented → eng-loop **PASS** (`scripts/eng_loop_best_ball_full_pitch.py`).

## Goal

**Best camera (ball)** is a **ball pick**, not a FOV crop.

- Video stage: **whole-pitch mosaic** (P10|P9 / P7|P8) with **all** player boxes + **one** orange ball
- Pitch 1: same multi-cam players + that one ball (yellow)
- Do **not** drop to a single-cam tile that hides other players

## Gate

```bash
python3 scripts/eng_loop_best_ball_full_pitch.py
```

All scores ≥ **9/10**. Report: `reports/eval_match3/improve_eng_loop/best_ball_full/scores.json`

## Components

| # | Done when |
|---|-----------|
| 01 | `Best camera (ball)` builds via `mosaic_quads_coach` (4 tiles) |
| 02 | Stage image height ≈ mosaic (not single big tile only) |
| 03 | Bag has players on ≥3 quads |
| 04 | Single orange ball mosaic-wide (`_keep_single_mosaic_ball`) |
| 05 | Pitch players ≥ 0.9 × Whole-pitch fused count (same frame) |
| 06 | Pitch ball present if Whole-pitch fuse has ball (same frame) |
| 07 | `used_cams` lists 4 quads (not 1 best cam) |
| 08 | Soft ball recover: if filter drops all balls, keep best mappable raw ball |
| 09 | Defish ON path unchanged |
| 10 | Unit/smoke PASS |

## Wire

`cams_for_view` / `build_cam_view`: Best camera → same as Whole pitch mosaic + single-ball.
Caption may still say best ball cam id.
