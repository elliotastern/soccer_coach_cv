# ball_postprocessing_test

Top Left `0:26–0:31` · P10 · ranking from prior train/gold studies.

| # | id | raw rate | emit hold | mean emit |
|---|---|---:|---:|---:|
| 1 | `baseline_topk2` | 50.7% | 15.3% | 0.838 |
| 2 | `topk3` | 50.7% | 15.3% | 0.838 |
| 3 | `hflip_tta` | 56.0% | 18.0% | 0.837 |
| 4 | `multiscale_1p5` | 50.7% | 15.3% | 0.838 |
| 5 | `sahi_fallback` | 78.0% | 26.7% | 0.832 |
| 6 | `thr50_topk2` | 38.7% | 15.3% | 0.838 |
| 7 | `emit80_pass` | 15.3% | 15.3% | 0.838 |
| 8 | `bytetrack_iou08` | 50.7% | 30.7% | 0.707 |
| 9 | `bytetrack_emit80` | 50.7% | 14.0% | 0.835 |
| 10 | `kalman_detect` | 72.0% | 15.3% | 0.838 |
