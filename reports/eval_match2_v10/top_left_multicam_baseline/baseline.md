# Match2 Top Left — 6 P-cam multicam baseline

Window 0:26–0:31 · cams `P1, P6, P7, P8, P10, P12` · v10 · no SAHI · detect@1920x1080

Proxy R/P: scored only when **selected cam is P10** (gold is P10-only).

## Baseline A — max_conf @0.30

- Selected frames: 295/300
- Selection share: `{"P1": 0.006666666666666667, "P10": 0.38333333333333336, "P12": 0.043333333333333335, "P6": 0.02, "P7": 0.51, "P8": 0.02, "none": 0.016666666666666666}`
- Mean cams with det: 2.00
- Proxy (P10-selected) P=0.948 R=0.948 n_frames=115 tp/fp/fn=109/6/6
- P10 single-cam @0.30 P=0.933 R=0.528

## Baseline B — emit ≥0.80 after max_conf

- Emitted: 47/300
- Proxy P=1.000 R=1.000 (n_frames=47)

Goal: R≥0.8 P≥0.9 (system). Proxy is biased toward P10-win frames.
