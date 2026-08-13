# Match 2 v10 video system

Checkpoint: `models/v10_snaps/post_train/checkpoint.pth`  
Stack: detect=0.3 → ByteTrack → emit=0.8 Kalman=True SAHI=False  
Master cams: Cam5plus + Cam4plus  
Strip: t=33.0s, 100 frames  
Gold warmup: 10 frames before each held-out gold JPEG

## Client primary — gold 50 with tracker warmup @ emit 0.80

- P_emit: **1.000**
- n_emitted: **27**
- FP: **0**  TP: 27  FN: 35
- hollow: False
- clear-ball R: 0.43548387096774194

## Product strip — Cam5plus/Cam4plus best-cam (no GT on full strip)

- n_emitted: **0** / 100 (rate 0.0)
- mean emit conf: None
- per cam: {'Cam5plus': 0, 'Cam4plus': 0}

P_emit/FPs come from held-out Match 2 gold 50 with video warmup + emit gate. The Cam5plus/Cam4plus strip reports product n_emitted (how often we publish). Strip has no full GT; overlapping gold frames are scored separately.
