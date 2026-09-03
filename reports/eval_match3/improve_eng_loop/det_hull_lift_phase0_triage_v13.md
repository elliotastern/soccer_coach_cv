# Detector + hull lift — post-v13 triage (2026-09-03)

Product fuse unchanged (F1+F2+F0+F3, EMIT_CONF=0.80, AGREE_M=4, MIN_SUPPORT=0.20).
Product ball checkpoint: `models/v13_residual_snaps/post_train/checkpoint.pth`.

## Scores

| Pack | Metric | v12 / prior | v13 |
|------|--------|------------:|----:|
| Strip P10 | clear_ball_R (product F0) | 0.879 | **0.888** |
| Strip P8 | clear_ball_R (product F0) | 0.846 | **0.904** |
| Holdout gallery | clear_ball_proxy_R | 0.892 | **0.893** (gate 0.884) |

Holdout: more clear proxies (241→298) with flat R — recall coverage widened, rate unchanged.

## Fuse vs map/det

Holdout clear FNs with another cam mapped ≥0.80: **0/32** (`holdout_other_cam_funnel_v13.json`). Not fuse-wiring.

## Holdout FN buckets (v13 caches)

| Bucket | n |
|--------|--:|
| `mapped_conf_below_emit` | 26 |
| `focus_map_fail` (mostly `P_Goal1\|low_support`) | 6 |

Focus conf on FNs: **21** in 0.50–0.79, 5 in 0.20–0.49, 6 ≥0.80 (map fails).

Systemic (≥2 caches): `mapped_conf_below_emit|P10`, `mapped_conf_below_emit|P6`.
Clip-only: Goal1 low_support / soft conf on `t01767.3s` — **do not hull**.

## Strip FN buckets (label clear, v13 caches)

| Pack | clear_R | Remaining |
|------|--------:|-----------|
| P10 | 0.991 | 1× soft conf |
| P8 | 0.915 | 16× soft conf, 7× no_det, 2× emit_fp |

## Branch decision (locked constraints)

- Landmarks / L2: **closed** (FOV fully labelled).
- Hull: still **skip** (no systemic `low_support` on ≥2 holdout caches).
- Next lever: **detector specialty v14** on remaining **P8** soft/no_det (+ tune-gallery soft FNs). **Do not train on holdout gallery frames.**
- Catch @1288 preferred when Tailscale returns; Mac @560 continue-from-v13 meanwhile.

## Eng-loop

`eng_loop_match3_improve.py`: all ≥9 except `l2_overlap=6.0` (P1/P6 RT>0.15) — expected with landmarks locked closed; not actionable.
