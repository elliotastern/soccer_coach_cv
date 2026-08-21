# Holdout random pack — clear-ball baseline

Frozen starts **2026-08-21**. Map-first pass applied after freeze (P10 hull + MIN_SUPPORT 0.20).

## Spec

| Field | Value |
|-------|--------|
| Role | **holdout** (honest gate / baseline) |
| Seed | **20260821** |
| Starts (s) | **166.0, 519.7, 552.8, 821.8, 1767.3** |
| Gap from tune | ≥25 s from `{627.9, 931.2, 1162.3, 1237.3, 1714.0}` |
| Checkpoint | `models/v12_hard_snaps/post_train/checkpoint.pth` |
| Gallery | `reports/eval_match3/pitchmap_gallery_holdout/` |
| Build | `scripts/gold_set/build_match3_holdout_random.py` |
| Fuse score | F1+F2+F0+F3 via `score_match3_ball_m1.py` pack `random_holdout` |

## Baseline → after map-first

| Stage | clear_ball_proxy_R | clear_emit / clear | Notes |
|-------|-------------------:|-------------------:|-------|
| Freeze (pre-hull) | **0.556** | 134 / 241 | Product F0, MIN_SUPPORT 0.25, no P10 hull |
| + P10 `hull_image_points` | **0.867** | 209 / 241 | Strip P_emit held |
| + MIN_SUPPORT **0.20** (H1 promote) | **0.884** | 213 / 241 | Strip P_emit held; agree_among_emit ~0.34 |

| Pack | Role | R (current) |
|------|------|------------:|
| `random` (tune) | report-only | 0.625 |
| **`random_holdout`** | **gate** | **0.884** |

Per holdout cache (post map-first): see `m1_provisional.json` → `packs.random_holdout.per_cache`.

## Spot-check / residual

- Spot: `reports/ball_testing/holdout_map_fn_spot/` — 73 P10 `low_support` FNs, in-FOV feet → hull OK.
- Residual FN: mostly clip `focus_map_fail` (22) vs conf (6). Other-cam ≥0.80 on clear FNs: **0/28** (`holdout_other_cam_funnel.json`).
- **Skipped** t552 one-clip re-detect (map still dominates).

## How to use

- **Holdout R ≥ 0.80** with strip P_emit ≥ 0.80 = clear-ball product-wide gate for this Match 3 pack.
- Do not retune hull on holdout times without a new frozen holdout seed.
