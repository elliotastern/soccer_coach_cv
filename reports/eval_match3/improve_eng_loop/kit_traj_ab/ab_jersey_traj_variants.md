# Jersey trajectory / history A/B vs 50/50

n_obs=5635 tracks=742 feat=`hard_center_50` iou=0.3

## Before (`per_frame`) vs best-by-score

| metric | before | after (`vote_last3`) |
|---|---:|---:|
| Team0 / Team1 | 61.9% / 38.1% | 61.6% / 38.4% |
| Off 50/50 | 11.8 pp | 11.6 pp |
| Track flip rate | 0.129 | 0.078 |
| balance min/max | 0.617 | 0.624 |

## Ranking (closest to 50/50 first)

| rank | variant | pct0 | pct1 | off50 | flips | bal | score |
|---:|---|---:|---:|---:|---:|---:|---:|
| 1 | `vote_last3` | 61.6 | 38.4 | 11.6 | 0.078 | 0.624 | 67.0 |
| 2 | `per_frame` | 61.9 | 38.1 | 11.8 | 0.129 | 0.617 | 64.8 |
| 3 | `vote_last5` | 62.0 | 38.0 | 12.0 | 0.052 | 0.612 | 66.6 |
| 4 | `vote_last8` | 62.1 | 37.9 | 12.1 | 0.040 | 0.611 | 66.9 |
| 5 | `sticky_0.70` | 62.2 | 37.8 | 12.2 | 0.120 | 0.607 | 64.1 |
| 6 | `sticky_0.85` | 62.3 | 37.7 | 12.3 | 0.119 | 0.605 | 64.0 |
| 7 | `vote5_sticky085` | 62.4 | 37.6 | 12.3 | 0.038 | 0.604 | 66.2 |
| 8 | `feat_ema_0.30` | 64.0 | 36.0 | 14.0 | 0.011 | 0.563 | 63.0 |
| 9 | `feat_ema_0.15` | 64.0 | 36.0 | 14.0 | 0.003 | 0.562 | 63.1 |
| 10 | `feat_median_5` | 64.8 | 35.2 | 14.8 | 0.017 | 0.542 | 60.6 |
| 11 | `feat_median_all` | 66.0 | 34.0 | 16.0 | 0.007 | 0.515 | 58.0 |
