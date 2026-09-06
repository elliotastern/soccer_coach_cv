# Fusion 8/10 → next 10 (no labeling / no user intervention)

**Baseline:** overall fusion ~8/10 (ball ~9, kit-on-fuse ~6.5–7.5, Match4 quad kit consensus ceiling ~7.7).  
**Constraint:** zero new human kit labels; no Catch power-on / P1–P6 batch; no merge/polarity dead-ends.

**Evidence used:** `kit_theories5_ab` (T1=70.2), `kit_undershirt_ideas10`, `loop_kit_balance`, `kit_fusion_gate_locked`, Ali `kit_hold`, DEAD_ENDS (ball specialty / merge soft / polarity).

## Ranked ideas (implement order)

| rank | id | idea | conf | est kit lift | evidence |
|---:|---|---|---:|---|---|
| 1 | N1 | **Productize annulus_zero_30** into `jersey_feature` (keep sticky/vote) | 9.5 | +0.5–1.0 | theories5 winner; product still center50-only |
| 2 | N2 | **Median-5 feature buffer** per track before assign | 9.0 | +0.3–0.7 | ideas10 #1 `median5_sticky` best score |
| 3 | N3 | **Fuse-then-color**: assign team on merged cluster from multi-cam crops | 8.5 | +0.5–1.0 | 52% frames cam share spread >0.35 |
| 4 | N4 | **Auto white seed bank** from dual-color center crops → freeze fit | 8.5 | +0.5–1.5 | replaces human kit-ref; dual≈23% of crops |
| 5 | N5 | **Hard-drop fisheye-edge** kit crops from centroid fit (not soft weight) | 8.0 | +0.2–0.5 | T2 soft P10 weight ≈noop; P10 still 74/26 |
| 6 | N6 | **Center-vs-edge veto**: outer≪center blue → center-only fracs | 8.0 | +0.2–0.4 | ideas10 #3/#8 high retain |
| 7 | N7 | **Gated dual_to_white**: only if center white≥blue; else sticky | 7.5 | cut false-white flicker | kit_hold dual thrash risk |
| 8 | N8 | **Birth wait**: unsure until age≥3 or ≥2 cams agree (no age lock flip) | 7.5 | fewer wrong births | T5 hard agelock failed; soft wait ≠ that |
| 9 | N9 | **Match3→Match4 auto centroid transfer** (HSV affine / hist match) | 7.0 | unlock M4 without labels | speculative; lighting risk |
| 10 | N10 | **Ball-proximal majority lock** for possession carrier team | 6.5 | event/team consistency | secondary; thin consensus lift |

## Do not put in the top 10

- Human kit-ref relabel (8503) — violates constraint  
- Catch P1/P6 fullcam batch — needs user/machine intervention; still the only path to Match4 consensus **9**  
- Soften `PLAYER_MERGE_*` / polarity flip / specialty ball RF-DETR — dead-ends  

## Suggested eng-loop

1. N1 + N2 on Match4 traj obs + Ali 15s re-render  
2. If share/flicker improve → N3 on fuse path  
3. N4 freeze A/B vs current online fit  
4. N5–N8 only if residual skew/flicker remains  
5. N9/N10 last (higher risk / lower EV)
