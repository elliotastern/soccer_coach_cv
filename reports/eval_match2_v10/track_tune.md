# Track enhancement tune (v10)

Tuned on **Match 2 train 87** (not gold). Locked pick, then scored held-out gold 50. Detect floor 0.3, 10-frame video warmup. Detector weights unchanged.

**Pick:** `detector_only`. No Kalman/ByteTrack setting beat raw RF-DETR on train without losing balls.

## Train 87 (124 GT balls)

| config | recall | precision | tp/fp/fn |
|---|---:|---:|---|
| **detector_only** | **0.952** | 0.975 | 118/3/6 |
| ByteTrack IoU 0.8 (legacy 80px or scaled) | 0.935 | 0.983 | 116/2/8 |
| ByteTrack IoU 0.5 | 0.831 | 0.981 | 103/2/21 |
| ByteTrack IoU 0.3 | 0.661 | 0.988 | 82/1/42 |
| ByteTrack IoU 0.3 + emit 0.80 | 0.323 | 1.000 | 40/0/84 |
| Kalman as detect | 0.556 | 0.793 | 69/18/55 |
| Kalman + ByteTrack IoU 0.3 | 0.250 | 0.861 | 31/5/93 |

## Held-out gold 50 (62 GT balls)

| config | recall | precision | tp/fp/fn |
|---|---:|---:|---|
| **detector_only** | **0.919** | 0.966 | 57/2/5 |
| ByteTrack IoU 0.8 | 0.903 | 0.966 | 56/2/6 |
| ByteTrack IoU 0.5 | 0.887 | 0.965 | 55/2/7 |
| Kalman as detect | 0.710 | 0.880 | 44/6/18 |
| ByteTrack + emit 0.80 | 0.403 | 1.000 | 25/0/37 |

Looser ByteTrack matching and Kalman both **lost balls**. Emit 0.80 is precision-only. Scaled 4K association radius did not recover the two dropped train boxes vs the old 80px cap.

JSON: `track_tune.json`.
