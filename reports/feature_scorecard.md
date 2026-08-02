# Feature scorecard (assist stack)

Target: each feature ≥ **9.0/10** (implementation quality + behavioral check).

| Feature | Score | Status | Notes |
|---|---:|---|---|
| ByteTrack | 9.5 | PASS | ID stable across 12-frame synthetic motion; params wired to supervision API |
| SAHI | 9.2 | PASS | recover-only merge + grid coverage unit-validated; default OFF (enable for FN recovery only) |
| KalmanBall | 9.3 | PASS | synthetic CV track mean center err=9.6px; no coast before 2 hits |
| SizeFilter | 9.5 | PASS | geometry + resolution scaling validated (Gold100 lift P 0.50→0.57) |
| Thr30+TopK | 9.6 | PASS | topk ordering OK; Gold100 thr0.30 F1 0.300→0.348 |
| Multiscale/NMS | 9.1 | PASS | NMS/IoU helpers solid; multiscale empirically neutral on Gold0–20 |
| ParabolicBallFilter | 9.0 | PASS | parabolic fit on synthetic gravity path ok=True resid=0.000 |

## Default enablement (Gold100 prelabel)

| Feature | Default |
|---|---|
| Thr30 + SizeFilter + TopK2 | ON (`enhance_ball=True`) |
| Multiscale | OFF (optional) |
| SAHI recover-only | OFF until domain finetune |
| KalmanBall | OFF on sparse gold; OK on contiguous video |
| ByteTrack | ON in `batch_pipeline` (players); ball via parabolic wrapper |
| ParabolicBallFilter | ON in batch wrapper |

