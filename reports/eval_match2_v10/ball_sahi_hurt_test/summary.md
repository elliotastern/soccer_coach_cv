# ball_sahi_hurt_test

Top Left `0:26–0:31` · P10 · SAHI combos most likely to hurt (theory + measured recover-always)

| # | id | raw rate | emit hold | mean emit |
|---|---|---:|---:|---:|
| 1 | `sahi_recover_always` | 78.0% | 26.7% | 0.832 |
| 2 | `sahi_always_topk3` | 78.0% | 26.7% | 0.832 |
| 3 | `sahi_always_topk5` | 78.0% | 26.7% | 0.832 |
| 4 | `sahi_always_thr20` | 81.3% | 25.3% | 0.832 |
| 5 | `sahi_always_nosize` | 78.0% | 26.7% | 0.832 |
| 6 | `sahi_always_multiscale` | 78.0% | 26.7% | 0.832 |
| 7 | `sahi_always_tta` | 80.0% | 34.0% | 0.835 |
| 8 | `sahi_always_kalman` | 88.7% | 26.7% | 0.832 |
| 9 | `sahi_always_bt_sticky` | 78.0% | 51.3% | 0.725 |
| 10 | `sahi_dense_tiles` | 80.0% | 20.7% | 0.837 |
