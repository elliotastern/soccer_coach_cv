# Match2 Top Left — 6 P-cam multicam baseline (dual gold)

Window 0:26–0:31 · cams `P1, P6, P7, P8, P10, P12` · v10 · no SAHI · detect@1920x1080

System score: when selected cam is **P7 or P10**, compare to that cam's gold.
Goal: **R≥0.8 P≥0.9** on the covered (P7∪P10-selected) set.

## Baseline A — max_conf @0.30

- Selected frames: 295/300
- Gold-covered share: 89.3% (268 frames)
- Selection share: `{"P1": 0.006666666666666667, "P10": 0.38333333333333336, "P12": 0.043333333333333335, "P6": 0.02, "P7": 0.51, "P8": 0.02, "none": 0.016666666666666666}`
- Mean cams with det: 2.00
- **P7∪P10 system** P=0.750 R=0.785 n_frames=268 tp/fp/fn=201/67/55 goal=MISS
- P10-selected only P=0.948 R=0.948 n_frames=115 tp/fp/fn=109/6/6 goal=HIT
- P7-selected only P=0.601 R=0.652 n_frames=153 tp/fp/fn=92/61/49 goal=MISS
- P10 single-cam @0.30 P=0.933 R=0.528
- P7 single-cam @0.30 P=0.528 R=0.535

## Baseline B — emit ≥0.80 after max_conf

- Emitted: 47/300
- **P7∪P10 system** P=1.000 R=1.000 n_frames=47 tp/fp/fn=47/0/0 goal=HIT
- P10-selected P=1.000 R=1.000 n_frames=47 tp/fp/fn=47/0/0 goal=HIT
- P7-selected P=0.000 R=0.000 n_frames=0 tp/fp/fn=0/0/0 goal=MISS

## Verdict vs goal

**MISS** live @0.30 on covered frames: R=0.785 (need ≥0.8), P=0.750 (need ≥0.9).
