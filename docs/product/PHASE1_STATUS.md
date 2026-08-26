# Phase 1 status (honest snapshot)

Canonical scope: [PHASE1_SCOPE.md](PHASE1_SCOPE.md). Pitch: [PITCH1_DIMENSIONS.md](PITCH1_DIMENSIONS.md) (not FIFA 105×68).  
Clear-ball: [MATCH3_CLEAR_BALL.md](MATCH3_CLEAR_BALL.md) · holdout: [`reports/ball_testing/HOLDOUT_BASELINE.md`](../reports/ball_testing/HOLDOUT_BASELINE.md).

Check clip (25 s match @ 30 fps, regular speed; mosaic + Pitch 1; defish tiles / no second undistort; pitch N-up / S-dn + `tile!=side · P8=north`): [`reports/eval_match3/improve_eng_loop/phase1_check/coach_mosaic_pitch_min.mp4`](../reports/eval_match3/improve_eng_loop/phase1_check/coach_mosaic_pitch_min.mp4).

**Phase 1 proof pack** (per-pillar clips + manifest): [`reports/eval_match3/improve_eng_loop/phase1_proof/manifest.json`](../reports/eval_match3/improve_eng_loop/phase1_proof/manifest.json). Rebuild: `python3 scripts/gold_set/build_phase1_proof_pack.py`.

**Client handover:** [PHASE1_CLIENT_HANDOVER.md](PHASE1_CLIENT_HANDOVER.md) · **Delivery manifest:** `python3 scripts/gold_set/build_phase1_delivery_manifest.py` → [`delivery_manifest.json`](../reports/eval_match3/improve_eng_loop/delivery_manifest.json).

## Scorecard (out of 10; **≥7 = enough for Phase 1**)

Updated after map-first pass (P10 hull + MIN_SUPPORT 0.20). **≥7 = pass.**

| Phase 1 requirement | /10 | Pass (≥7)? | Notes |
|---|---:|:---:|---|
| Ball detect + map (precision-first) | **8** | Yes | Strip product-F0 **P_emit** P10 1.0 / P8 0.966. Mosaic ball still solid when present. |
| Clear-ball coverage (R ≥ 0.80 product-wide) | **8** | Yes | **Holdout proxy 0.884** (was 0.556 freeze) after P10 hull + MIN_SUPPORT 0.20. Strips product-F0 clear_R pass. Tune pack still 0.625 (report-only). |
| Player boxes on multi-cam video | **8** | Yes | Unchanged — green boxes on mosaic cams. |
| Pitch 1 mapping (meters, not FIFA) | **7** | Yes | Unchanged — usable; some churn. |
| Team A/B color ID (Match 3) | **8** | Yes | `kit_mode=match3` eng-loop PASS; blue/white hard rules + hue lock. |
| Team A/B color ID (Match 4) | **8** | Yes | `kit_mode=auto` + fuse cap; 90s eng-loop **PASS** (collapse 7% vs 58% baseline). |
| Heuristic events (all 5 types) | **8** | Yes* | Eng-loops **PASS**: fuse 15s **P_emit 1.0** + holdout pass window; synth **P=1.0** all types; carrier pid gold **42**. *Full-match batch proof still open.* |
| Review app (Streamlit coach view) | **8** | Yes | Unchanged. |
| Batch / checkpoints / export | **6** | No | Match 4 5‑min quad **done** on Catch; **full** P10+P1 batch running in tmux `match4_full`. Manifest: `delivery_manifest.json` (smoke passes; full-match export still in progress). |
| Process 2 matches + 3rd handover | **3** | No | Unchanged. |
| Commercial-safe stack (no YOLO / no FIFA as product) | **9** | Yes | Unchanged. |

**Phase 1 overall: ~7.8/10** — ball + holdout clear-ball + events at the bar; team ID unified; 2-match delivery still open.

## Next (see [PHASE1_SCOPE.md](PHASE1_SCOPE.md) § “What needs to be done next”)

1. Run **batch pipeline** on **2 full matches** with checkpoints + CSV/JSON export.  
2. **Handover session** on a **3rd match**.  
3. Optional map polish (P8 upper / P9 edge) before locking export quality.

Ball precision, **holdout clear-ball (0.884)**, and heuristic events (**fuse 15s P_emit 1.0** on pass/dribble; synth **P=1.0** all five types) pass ≥7. Remaining blocker: full-match delivery.

## Evidence (events — rerate 2026-08-24, `team_core`)

Report: [`EVENT_ACCURACY_RERATE_2026-08-24.json`](../reports/events_testing/EVENT_ACCURACY_RERATE_2026-08-24.json)

| Type | Product fuse 15s (stride 4) | Synth gold |
|------|---------------------------:|------------:|
| **pass** | P **1.0** (2/2) | P **1.0** (2/2) |
| **dribble** | P **1.0** (1/1) | P **1.0** (1/1) |
| **shot** | 0 emits, 0 FP | P **1.0** (1/1) |
| **recovery** | 0 emits, 0 FP | P **1.0** (1/1) |
| **movement** | 0 emits (no gold in clip) | P **1.0** (1/1) |

check25_human at stride 15 remains **report-only** (coarse sample). Holdout `real_fuse_holdout_pass` scores pass outside handover window.

**Batch audit (2026-08-25):** [`BATCH_EVENTS_AUDIT_20260825.json`](../reports/events_testing/BATCH_EVENTS_AUDIT_20260825.json) — Match 4 quad 5‑min: **0 dribble**, pass-only single-cam emits; fuse eval @20s shows movement vs single-cam pass cluster (report-only).

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

**Demo-ready** Match 3 review for ball precision, players, team colors, and heuristic events (all 5 types on eng-loop gold + batch wiring).  
Still open: 2-match delivery.
