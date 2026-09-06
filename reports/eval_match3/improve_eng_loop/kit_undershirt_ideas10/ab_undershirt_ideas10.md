# 10 undershirt jersey ideas — tested

Match4 real torsos n=600; stress = opposite-color outer ring (0.24).

**Winner: `median5_sticky`**

| rank | # | idea | score | share | off50 | retain | flips |
|---:|---:|---|---:|---:|---:|---:|---:|
| 1 | 9 | `median5_sticky` | 88.8 | 49.5/50.5 | 0.5 | 0.645 | 0.045 |
| 2 | 10 | `dual_soft_vote5_bal` | 88.7 | 49.5/50.5 | 0.5 | 0.713 | 0.139 |
| 3 | 2 | `annulus_zero_30` | 88.2 | 50.3/49.7 | 0.3 | 0.998 | 0.493 |
| 4 | 1 | `center50_only` | 86.9 | 49.3/50.7 | 0.7 | 0.998 | 0.507 |
| 5 | 6 | `dual_soft_white` | 86.9 | 49.3/50.7 | 0.7 | 0.998 | 0.507 |
| 6 | 3 | `center_vs_edge_vote` | 84.1 | 48.5/51.5 | 1.5 | 0.995 | 0.500 |
| 7 | 8 | `edge_blue_ignore` | 83.9 | 48.0/52.0 | 2.0 | 0.989 | 0.473 |
| 8 | 5 | `outer_highsat_down` | 66.5 | 44.4/55.6 | 5.6 | 0.595 | 0.489 |
| 9 | 4 | `low_sat_body` | 60.0 | 36.2/63.8 | 13.8 | 0.998 | 0.433 |
| 10 | 7 | `dual_unsure_vote5` | 52.1 | 69.4/30.6 | 19.4 | 0.763 | 0.057 |

## Ideas

1. **center50_only** — Sample only center 50×50% (ignore sleeve/collar).
2. **annulus_zero_30** — Zero-weight outer 30% ring (undershirt zone).
3. **center_vs_edge_vote** — Center white-dom + edge blue-dom → white kit.
4. **low_sat_body** — Up-weight low-S pixels (white body vs saturated sleeve).
5. **outer_highsat_down** — Down-weight high-S only in outer third.
6. **dual_soft_white** — Both blue&white ≥0.35 on center → white @0.70.
7. **dual_unsure_vote5** — Dual-color → unsure; vote5 fills identity.
8. **edge_blue_ignore** — If outer much bluer than center, use center-only fracs.
9. **median5_sticky** — Center50 + sticky hold (anti-flicker).
10. **dual_soft_vote5_bal** — Center50 + dual-soft + vote5 + soft 50/50 nudge.
