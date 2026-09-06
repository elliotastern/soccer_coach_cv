# Jersey undershirt variant A/B

n_crops=600 ring=0.24

| rank | variant | score | retain | flip | drift | sep | bal | cover |
|---:|---|---:|---:|---:|---:|---:|---:|---:|
| 1 | `hard_center_50` | 94.8 | 1.000 | 0.000 | 0.012 | 0.982 | 0.625 | 0.962 |
| 2 | `inner_ellipse_55` | 94.5 | 1.000 | 0.000 | 0.011 | 0.996 | 0.614 | 0.950 |
| 3 | `top50_center_w` | 93.2 | 0.996 | 0.004 | 0.058 | 0.952 | 0.575 | 0.940 |
| 4 | `gauss_sat_gate` | 93.2 | 0.983 | 0.017 | 0.062 | 0.988 | 0.619 | 0.943 |
| 5 | `hard_center_35` | 92.8 | 1.000 | 0.000 | 0.007 | 1.054 | 0.635 | 0.818 |
| 6 | `raised_cosine` | 92.1 | 0.995 | 0.005 | 0.063 | 0.972 | 0.583 | 0.863 |
| 7 | `gauss_hue_trim` | 90.9 | 0.984 | 0.016 | 0.061 | 0.992 | 0.481 | 0.900 |
| 8 | `sleeve_x_gauss` | 90.7 | 0.974 | 0.026 | 0.172 | 0.899 | 0.540 | 0.938 |
| 9 | `gauss_soft` | 90.2 | 0.951 | 0.049 | 0.189 | 0.896 | 0.560 | 0.970 |
| 10 | `gauss_tight` | 89.9 | 1.000 | 0.000 | 0.009 | 1.015 | 0.652 | 0.608 |
| 11 | `baseline` | 50.2 | 0.186 | 0.814 | 0.598 | 0.716 | 0.557 | 1.000 |

## Before vs after (winner = `hard_center_50`)

Problem: outer kit / undershirt can disagree (e.g. white jersey + blue sleeves), so flat
pixel means in `jersey_feature` pull white-kit players toward the blue centroid.

Method: 600 real Match 4 torso crops from `m4_90s_det_plain.json`; stress = paint opposite
color on the outer ring; score retain/flip/drift/sep/balance/coverage.

| metric | before (baseline) | after (`hard_center_50`) |
|---|---:|---:|
| Team 0 / Team 1 share | 64.2% / 35.8% | 61.5% / 38.5% |
| Off 50/50 | 14.2 pp | 11.5 pp |
| Undershirt retain | 18.6% | **100%** |
| Flip under stress | 81.4% | **0%** |
| Kit-dim drift | 0.598 | 0.012 |
| Blue–white sep | 0.716 | 0.982 |
| Coverage | 100% | 96.2% |

Takeaway: center 50%×50% crop almost eliminates undershirt flips; team share only moves
slightly closer to even (~2.7 pp). Soft Gaussians help less; `gauss_tight` is pure but
drops too many crops (cover 0.61). Product `jersey_feature` not switched yet — report-only.
