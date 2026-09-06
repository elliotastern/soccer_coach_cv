# Jersey eng-loop theories (≥8/10 confidence)

Current best product feel: **~6.5/10** (`kit_hold` + center/dual/sticky).
Evidence bases: Ali `meta_v16_kit_hold`, Match4 `m4_traj_obs` (n=5635), ideas10 A/B.

## Locked diagnosis

- **23%** of Match4 center crops are dual-color (blue&white ≥0.35) — undershirt / mixed torso.
- **P10** detections are **73%** blue>white (P8 ≈51%).
- **52%** of Match4 frames have cam-to-cam team-share spread **>0.35** — cameras disagree hard.
- Ali `kit_hold` still has **16%** of frames with |share−0.5|>0.15 (worst **12 vs 3**).
- ideas10: **annulus/center** ≈ perfect undershirt retain; **sticky** ≈ lowest flips — **never stacked**.

## Five theories (≥8/10 each)

### T1 — Stack annulus (or center50) sampling + sticky track lock (conf **9/10**)
**Why:** Separately, annulus_zero_30 retain≈1.00 @ ~50/50; median5_sticky flips≈0.05. Current best trades one for the other.  
**Do:** `annulus_zero_30` (or hard_center_50) features → assign → sticky/vote lock (no mid-clip centroid swap).  
**Expect:** keep ~50/50 + cut Ali frame thrash without losing undershirt retain.

### T2 — Downweight P10 (and fisheye-edge) in fit + multi-cam team vote (conf **9/10**)
**Why:** P10 alone is 74/26 blue-heavy; 52% of frames show large cam share disagreement. P8 is already ~50/50.  
**Do:** inverse-cam or `w_P10≤0.5` in centroid fit; fused team = weighted vote favoring image-center / P8/P9 over P10 periphery.  
**Expect:** fewer false blues from fisheye edge kits; fused map closer to even.

### T3 — Freeze kit-ref centroids; expand kit-ref with undershirt dual samples (conf **9/10**)
**Why:** Product kit-consensus path already requires freeze; online rebalance/EMA flipped whole teams (flicker). Kit-ref today is only **7/15** samples and was built pre-center/undershirt rules.  
**Do:** label 5–10 clear chest + 5 dual/undershirt whites on 8503; seed match-level `team_centroids.json`; disable session rebalance.  
**Expect:** stable polarity + better white recall on undershirt players.

### T4 — Multi-cam agreement gate before accepting a team flip (conf **8.5/10**)
**Why:** Half of frames have cams disagreeing on share; sticky alone still allows single-cam noise to enter fuse.  
**Do:** if ≥2 cams on a cluster agree → take that team; else keep prior sticky team (don’t flip on one cam).  
**Expect:** lower per-player flicker on mosaic/pitch dots.

### T5 — Fused roster prior: balance only births / low-conf, never age≥2 locks (conf **8/10**)
**Why:** Ali frames still hit 12–3; global equalize caused flicker; birth-only nudge in `kit_hold` moved share to 55/45 without dual-level thrash.  
**Do:** strengthen birth/ambiguous nudge toward equal fused counts; **hard lock** tracks with age≥2 (flip only on multi-cam agree streak).  
**Expect:** fewer 12–3 frames while preserving identity continuity.

## Explicitly weaker / not in the five

- low_sat_body / outer_highsat alone (ideas10: skew or poor retain)
- dual_unsure-only (collapsed to 69/31)
- chasing 50/50 by recoloring locked tracks every frame

## Next implement order

1. T1 stack (code-only, fast A/B + Ali re-render)  
2. T2 cam weights (A/B on traj obs)  
3. T4 agree-gate (fuse path)  
4. T5 birth lock harden  
5. T3 kit-ref relabel (human, then freeze)

## A/B results (2026-09-05)

Match4 theory obs n=2899, tracks=528. Ranking by composite score:

1. **T1_annulus_sticky** — **70.2** (best)
2. T3_freeze_first30 — 69.5
3. baseline_center_sticky — 69.5
4. T4_multicam_agree — 69.5 (≈ baseline on this harness)
5. T2_p10_downweight — 69.4
6. T5_birth_agelock — 62.1 (worse)

See `kit_theories5_ab/ab_theories5.md`.
