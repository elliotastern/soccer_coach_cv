# Match2 Top Left — soft 2-cam consensus vs baseline (dual gold)

Consensus: thr=0.15, min_cams=2, max_conf among supporters, no SAHI.

| stack | system P | system R | ΔP | ΔR | n_covered | mean cams w/ det | goal |
|---|---:|---:|---:|---:|---:|---:|---|
| baseline_a @0.30 | 0.750 | 0.785 | — | — | 268 | 2.00 | MISS |
| soft_consensus | 0.744 | 0.776 | -0.006 | -0.009 | 270 | 3.81 | MISS |

Selection share (consensus): `{"P1": 0.006666666666666667, "P10": 0.38333333333333336, "P12": 0.043333333333333335, "P6": 0.02666666666666667, "P7": 0.5166666666666667, "P8": 0.02, "none": 0.0033333333333333335}`

## Gate

NO-GO on both R and P for covered system — diagnose P7 vs P10 split.

Epipolar still blocked (no Match 2 extrinsics). Dense SAHI stays out of the live path.
