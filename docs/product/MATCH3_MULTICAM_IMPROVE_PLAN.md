# Match 3 multi-cam ball — improve loop plan

Engineering loop for **Pitch 1 / Field 1** ball `(x, y)` across the eight Match 3 cameras. Builds on [MATCH3_XY_BALL_PLAN.md](MATCH3_XY_BALL_PLAN.md) and the gallery funnel diagnosis (P1-dominated solo emits; quads starved by thr / hull / dets).

## Ball goals (do not relax)

| Role | Metric | Target |
|------|--------|--------|
| **Primary** | Precision of **emitted** ball outputs (`P_emit`) | **≥ 0.80** |
| **Secondary** | Clear-ball recall (near/clear view; start side ≥ 25 px) | **≥ 0.80** |
| **System (Match 3 fuse)** | When ≥2 cams map and agree ≤ 4 m: combined conf | **≥ 0.80** to emit |
| **Not the bar** | All-frame recall @ conf 0.80 on tiny/far balls | Stretch / Phase 2 |

Emit gate stays **conf ≥ 0.80**. Below that → do nothing. Do **not** average cams that disagree by more than **4 m**. Do **not** start Phase 2 temporal fusion.

Clear-ball R and `P_emit` are measured on labeled strips (Gold / agreed Match 3 clips), not on train-pack AP50.

## Why current gallery fails the goals

On 750 processed frames (5×150): emit rate ~13%, almost all **P1 solo**; only **3** two-cam agree frames. Quads are in the pool but die on Match-2 **P7 thr 0.60**, conf &lt; 0.30 on P8/P10, and **hull support** after tight 3–4 click strips. Pitch 1 meters (53.90×34.84) are already correct.

## Loop steps (precision-first)

| Step | Do | Done when | Goal link |
|------|----|-----------|-----------|
| **T1** | Match-3 detect thr: all cams **0.30** (drop Match-2 P7@0.60 on this path only) | Gallery / fuse uses `MATCH3_THR_BY_CAM`; unit/eng-loop green | More dual-maps without lowering emit 0.80 |
| **L1** | Restore **≥4** clicks on P8, P9, P_Goal1 (post or box corner; no FIFA penalty invent) | `manual_clicks` DLT, round-trip ≤ 0.15 m | Honest H + larger hull |
| **L2** | Add **overlapping** landmarks P1↔P6 (e.g. other circle / opposite box if visible) | Each of P1/P6 has ≥5 marks spanning both sides of play | Agree where ball actually is |
| **H1** | Optional: `MIN_SUPPORT` **0.20–0.25** A/B (not 0) | Same labeled strip: `P_emit` does not fall below 0.80 | Clear-ball R↑ without ghost xy |
| **C1** | Quad-focused eval clips (ball in P8/P9/P10 FOV) | ≥1 clip per quad with thr-pass dets | Measure mapping, not only P1 plays |
| **E1** | Re-render Match 3 pitchmap gallery; report solo / agree / emit rate | Manifest updated; agree ≫ 3 on same seeds if L2 done | Coverage toward clear-ball R |
| **M1** | Score `P_emit` + clear-ball R on agreed strip | Strip pack `match3_quad_p10_31` + editable `/match3-m1` review; provisional auto-QA 39/39; **P_emit ≥ 0.80**, clear-ball **R ≥ 0.80** after human confirm | Acceptance |

## Hard no

- Widen agree beyond 4 m or average distant cams
- Remove hull gate entirely without L2
- Replace emit 0.80 with lower solo thr
- FIFA 105×68 / invent penalty spots on Pitch 1
- Phase 2 RNN / occlusion-merge as a Phase 1 requirement

## Wire

- Thr: `scripts/gold_set/demo_locked_oos_pitchmap.py` + Match 3 gallery (not Match 2 `TOP_LEFT_THR_BY_CAM` globally)
- Calib: `scripts/gold_set/match3_landmarks.py`, marker UI
- Map/fuse: `src/mapping/match3_xy.py`
- Pitch meters: `docs/product/PITCH1_DIMENSIONS.json`
- Eng-loop: `scripts/gold_set/eng_loop_match3_improve.py` (plan ≥ 9/10)
- M1 strip: [MATCH3_M1_STRIP.md](MATCH3_M1_STRIP.md) (`match3_quad_p10_31`)

## Checks

- Eng-loop plan score **≥ 9/10**
- Landmark round-trip ≤ 0.15 m after L1/L2
- Fuse: 2-cam ≤4 m → median; disagree → no midpoint; solo conf 0.4 → no emit
- Camera id = video title (P9-004 → P9)
- Final: **P_emit ≥ 0.80**, clear-ball **R ≥ 0.80**
