# Phase 1 status (honest snapshot)

Canonical scope: [PHASE1_SCOPE.md](PHASE1_SCOPE.md). Pitch: [PITCH1_DIMENSIONS.md](PITCH1_DIMENSIONS.md) (not FIFA 105×68).

## Where we are vs Phase 1 final goal

| Area | Phase 1 final goal | Where we are now |
|---|---|---|
| Ball emit | **P_emit ≥ 0.80** @ IoU 0.5 (product emit gate) | Looking **decent** on Match 3 review / test strips. Full Gold100 (and agreed demo) PoC is still the acceptance gate. |
| Clear-ball recall | ≥ **0.80** (secondary coverage) | Lifted with Match 3 **defish** on P7–P10; random-pack / some strips still **below** full PoC. |
| Players on video + Pitch 1 | Precision-first bodies + mapped dots | Bodies **decent** on mosaic; boxes restored on all cams. Pitch map still shows **end-line doubles / hold ghosts**. |
| Team ID | Team A/B via color clustering | Coach ~**7.5–8/10**: demo-stable colors (no old flicker/swap); **not** “trust every dot” on tough crops / count churn. |
| Heuristic events | pass / dribble / movement / recovery / shot | Not the current review focus. |
| Delivery | Process **2 full matches** + handover on a **3rd** | Batch pipeline exists; full two-match client delivery not closed out here. |
| Explicitly **not** Phase 1 | Multi-cam occlusion fusion, learned ReID, FIFA pitch | Still out of scope. |

## Bottom line

**Demo-ready** for Match 3 review: ball, players, and team colors on Pitch 1 without the old team flicker/swap.

**Not final Phase 1 PoC yet:** still need emit/clear-ball gates on Gold, full-match batch delivery, and heuristic events — not end-line fuse polish alone.
