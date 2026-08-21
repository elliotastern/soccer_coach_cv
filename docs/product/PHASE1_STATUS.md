# Phase 1 status (honest snapshot)

Canonical scope: [PHASE1_SCOPE.md](PHASE1_SCOPE.md). Pitch: [PITCH1_DIMENSIONS.md](PITCH1_DIMENSIONS.md) (not FIFA 105×68).  
Clear-ball: [MATCH3_CLEAR_BALL.md](MATCH3_CLEAR_BALL.md) · holdout: [`reports/ball_testing/HOLDOUT_BASELINE.md`](../reports/ball_testing/HOLDOUT_BASELINE.md).

Check clip (30 s mosaic + Pitch 1): [`reports/eval_match3/improve_eng_loop/phase1_check/coach_mosaic_pitch_min.mp4`](../reports/eval_match3/improve_eng_loop/phase1_check/coach_mosaic_pitch_min.mp4).

## Scorecard (out of 10; **≥7 = enough for Phase 1**)

Updated after map-first pass (P10 hull + MIN_SUPPORT 0.20). **≥7 = pass.**

| Phase 1 requirement | /10 | Pass (≥7)? | Notes |
|---|---:|:---:|---|
| Ball detect + map (precision-first) | **8** | Yes | Strip product-F0 **P_emit** P10 1.0 / P8 0.966. Mosaic ball still solid when present. |
| Clear-ball coverage (R ≥ 0.80 product-wide) | **8** | Yes | **Holdout proxy 0.884** (was 0.556 freeze) after P10 hull + MIN_SUPPORT 0.20. Strips product-F0 clear_R pass. Tune pack still 0.625 (report-only). |
| Player boxes on multi-cam video | **8** | Yes | Unchanged — green boxes on mosaic cams. |
| Pitch 1 mapping (meters, not FIFA) | **7** | Yes | Unchanged — usable; some churn. |
| Team A/B color ID | **7** | Yes (barely) | Unchanged — demo OK, not trust-every-dot. |
| Heuristic events (pass/dribble/…) | **2** | No | Unchanged — not in path. |
| Review app (Streamlit coach view) | **8** | Yes | Unchanged. |
| Batch / checkpoints / export | **6** | No | Unchanged — not proven by delivery. |
| Process 2 matches + 3rd handover | **3** | No | Unchanged. |
| Commercial-safe stack (no YOLO / no FIFA as product) | **9** | Yes | Unchanged. |

**Phase 1 overall: ~7/10** — ball precision + holdout clear-ball at the bar; events / 2-match delivery still open.

Ball precision and **holdout clear-ball (0.884)** pass ≥7. Remaining blockers: heuristic events and full-match delivery.

## Evidence (ball)

| Metric | Value |
|--------|------:|
| Tune random proxy R (F0) | 0.625 |
| **Holdout** proxy R (F0) | **0.884** (freeze 0.556) |
| P10 strip product-F0 P / R | 1.0 / 0.876 |
| P8 strip product-F0 P / R | 0.966 / 0.801 |
| MIN_SUPPORT | 0.20 (H1 promote) |
| Holdout residual FN | map 22 / conf 6; other-cam ≥0.80 = 0/28 |

## Bottom line

**Demo-ready** Match 3 review for ball precision, players, team colors.  
Ball emit + **holdout clear-ball** at the Phase 1 bar. Still open: heuristic events and 2-match delivery.
