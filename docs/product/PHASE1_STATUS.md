# Phase 1 status (honest snapshot)

Canonical scope: [PHASE1_SCOPE.md](PHASE1_SCOPE.md). Pitch: [PITCH1_DIMENSIONS.md](PITCH1_DIMENSIONS.md) (not FIFA 105×68).

Check clip (30 s mosaic + Pitch 1): [`reports/eval_match3/improve_eng_loop/phase1_check/coach_mosaic_pitch_min.mp4`](../reports/eval_match3/improve_eng_loop/phase1_check/coach_mosaic_pitch_min.mp4)  
(fr 2390–4189 @ Match 3 60 fps, detect stride 15, playback 4 fps).

## Scorecard (out of 10; **≥7 = enough for Phase 1**)

Scored from the 30 s check clip + known PoC strip / random-pack numbers.

| Phase 1 requirement | /10 | Pass (≥7)? | Notes |
|---|---:|:---:|---|
| Ball detect + map (precision-first) | **8** | Yes | Ball on mosaic ~85% of clip samples; orange box + yellow pitch/trail when present. Strip P_emit often ≥0.80. |
| Clear-ball coverage (R ≥ 0.80 product-wide) | **6** | No | Clip looks OK locally; random-pack clear_R still ~0.53 — not Phase 1 done. |
| Player boxes on multi-cam video | **8** | Yes | Green boxes on P10/P9/P7/P8 across start/mid/end. |
| Pitch 1 mapping (meters, not FIFA) | **7** | Yes | Top-down Pitch 1 usable; some count/position churn mid-clip. |
| Team A/B color ID | **7** | Yes (barely) | Both kits most frames (94/120); still share swings and blue/red imbalance. Demo OK, not trust-every-dot. |
| Heuristic events (pass/dribble/…) | **2** | No | Not in this product path / clip. |
| Review app (Streamlit coach view) | **8** | Yes | Mosaic + Pitch 1 *is* the Phase 1 review surface. |
| Batch / checkpoints / export | **6** | No | Pipeline exists; not proven by this clip or closed 2-match delivery. |
| Process 2 matches + 3rd handover | **3** | No | Delivery acceptance not met. |
| Commercial-safe stack (no YOLO / no FIFA as product) | **9** | Yes | RF-DETR + Pitch 1 + color teams as designed. |

**Phase 1 overall: ~6.5/10 — not a full pass yet.**

At/above the bar: ball precision on strips, player boxes, Pitch 1 map, team demo, review UI.  
Blocking: clear-ball product-wide, events, and full-match delivery.

## Where we are vs Phase 1 final goal

| Area | Phase 1 final goal | Where we are now |
|---|---|---|
| Ball emit | **P_emit ≥ 0.80** @ IoU 0.5 (product emit gate) | Looking **decent** on Match 3 review / test strips. Full Gold100 (and agreed demo) PoC is still the acceptance gate. |
| Clear-ball recall | ≥ **0.80** (secondary coverage) | Lifted with Match 3 **defish** on P7–P10; random-pack / some strips still **below** full PoC. |
| Players on video + Pitch 1 | Precision-first bodies + mapped dots | Bodies **decent** on mosaic; boxes restored on all cams. Pitch map improved (goal-box hygiene); still not perfect. |
| Team ID | Team A/B via color clustering | Coach ~**7/10**: demo-stable colors; **not** “trust every dot”. |
| Heuristic events | pass / dribble / movement / recovery / shot | Not the current review focus. |
| Delivery | Process **2 full matches** + handover on a **3rd** | Batch pipeline exists; full two-match client delivery not closed out here. |
| Explicitly **not** Phase 1 | Multi-cam occlusion fusion, learned ReID, FIFA pitch | Still out of scope. |

## Bottom line

**Demo-ready** for Match 3 review: ball, players, and team colors on Pitch 1 without the old team flicker/swap.

**Not final Phase 1 PoC yet:** still need emit/clear-ball gates locked product-wide, full-match batch delivery, and heuristic events.
