# Match2 Top Left — soft 2-cam consensus vs baseline

Consensus: thr=0.15, min_cams=2, max_conf among supporters, no SAHI.

| stack | proxy P | proxy R | ΔP vs A | ΔR vs A | n_proxy_frames | mean cams w/ det |
|---|---:|---:|---:|---:|---:|---:|
| baseline_a @0.30 | 0.948 | 0.948 | — | — | 115 | 2.00 |
| soft_consensus | 0.948 | 0.948 | +0.000 | +0.000 | 115 | 3.81 |

Selection share (consensus): `{"P1": 0.006666666666666667, "P10": 0.38333333333333336, "P12": 0.043333333333333335, "P6": 0.02666666666666667, "P7": 0.5166666666666667, "P8": 0.02, "none": 0.0033333333333333335}`

## Gate

GO: proxy hit R/P goals on P10-selected frames → next is 5090 latency. Still need broader multi-cam GT before claiming full system.

Epipolar still blocked (no Match 2 extrinsics). Dense SAHI stays out of the live path.

## Decision (this loop)

| Question | Answer |
|---|---|
| Proxy R≥0.8 and P≥0.9 when P10 is selected? | **Yes** (~0.95 / 0.95) for baseline A and soft consensus |
| Soft consensus lift vs baseline A? | **None** on proxy (same P10-selected set); mean cams with det 2.00 → 3.81 |
| Who wins max_conf most? | **P7 ~51%**, P10 ~38% — proxy ignores most frames |
| Next | **5090 latency** for 6× detect; for true system R/P add **Cam4plus/P7 labels** or **calib→epipolar**. Do not chase dense SAHI for live. |

