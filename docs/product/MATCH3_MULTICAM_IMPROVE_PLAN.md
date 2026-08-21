# Match 3 multi-cam ball — improve loop plan

Engineering loop for **Pitch 1 / Field 1** ball `(x, y)` across the eight Match 3 cameras. Builds on [MATCH3_XY_BALL_PLAN.md](MATCH3_XY_BALL_PLAN.md) and the gallery funnel diagnosis (P1-dominated solo emits; quads starved by thr / hull / dets).

## Ball goals (do not relax)

| Role | Metric | Target |
|------|--------|--------|
| **Primary** | Precision of **emitted** ball outputs (`P_emit`) | **≥ 0.80** |
| **Secondary** | Clear-ball recall (near/clear view; start side ≥ 25 px) | **≥ 0.80** |
| **System (Match 3 fuse)** | When ≥2 cams map and agree ≤ 4 m: combined conf | **≥ 0.80** to emit |
| **Not the bar** | All-frame recall @ conf 0.80 on tiny/far balls | Stretch / Phase 2 |

Emit gate stays **conf ≥ 0.80**. Below that → do nothing. Do not average cams that disagree by more than **4 m** (no midpoint). Do **not** start Phase 2 temporal fusion.

Optional per-cam `hull_image_points` may expand support FOV without refitting landmark H (used on P_Goal1 for midfield balls outside the goal-box clicks).

Clear-ball R and `P_emit` are measured on labeled strips (Gold / agreed Match 3 clips), not on train-pack AP50.

## Why current gallery fails the goals

On 750 processed frames (5×150): emit rate ~13%, almost all **P1 solo**; only **3** two-cam agree frames. Quads are in the pool but die on Match-2 **P7 thr 0.60**, conf &lt; 0.30 on P8/P10, and **hull support** after tight 3–4 click strips. Pitch 1 meters (53.90×34.84) are already correct.

## Loop steps (precision-first)

| Step | Do | Done when | Goal link |
|------|----|-----------|-----------|
| **D0** | **Defish P7–P10:** fisheye tags, defished landmarks, `map_ball_box` undistort | A/B + gallery: agree **3→54**, clear proxy R **0.30→0.63** on random 5; P10 M1 **P_emit 1.0**, **clear_R 0.81** | Correct fisheye geometry; recall/agree lift — see [MATCH3_DEFISH.md](MATCH3_DEFISH.md) |
| **C1** | **Quad FN audit** + `hull_image_points` on P8/P9 + rebuild P8 strip gold | `fn_audit_match3_quad.py` → `c1_fn_audit.json`; P8 strip **P_emit 1.0**, F0 **clear_R 0.88**; quad proxy R **0.21→0.54** | Unlock quad maps without refitting H; P9 t00559 still det-limited |
| **C2** | **Quad det funnel:** v10 vs **v12_hard** (+ SAHI A/B) on quad caches | `funnel_match3_quad_det.py` → promote **v12_plain**; quad proxy R **0.54→0.58**; both M1 strips pass **P≥0.90 / R≥0.80** (F0) | P9 t00559 clip still ~0.28 proxy |
| **C3** | **P6 hull expand** (near-touch `hull_image_points`) — FN on P9 t00559 was P6 conf≥0.80 low_support | t00559 proxy **0.28→0.94**; quad pack **0.58→0.84**; both M1 strips still **P≥0.90 / R≥0.80** | Random gallery proxy still ~0.53 |
| **T1** | Match-3 detect thr: all cams **0.20** (was 0.30; no Match-2 P7@0.60) | M1 strip: P_emit stays 1.0; weak dual-maps unlock; galleries emit flat | More dual-maps without lowering emit 0.80 |
| **L1** | Restore **≥4** clicks on P8, P9, P_Goal1 (post or box corner; no FIFA penalty invent) | `manual_clicks` DLT, round-trip ≤ 0.15 m | Honest H + larger hull |
| **L2** | Add **overlapping** landmarks P1↔P6 (e.g. other circle / opposite box if visible) | Each of P1/P6 has ≥5 marks spanning both sides of play | Agree where ball actually is |
| **H1** | `MIN_SUPPORT` **0.20** (promoted after holdout A/B 0.867→0.884; strip P held) | Locked in `match3_xy.py`; was 0.25 | Clear-ball R↑ without ghost xy |
| **C1** | Quad-focused eval clips (ball in P8/P9/P10 FOV) | ≥1 clip per quad with thr-pass dets | Measure mapping, not only P1 plays |
| **E1** | Re-render Match 3 pitchmap gallery; report solo / agree / emit rate | Manifest updated; agree ≫ 3 on same seeds if L2 done | Coverage toward clear-ball R |
| **M1** | Score `P_emit` + clear-ball R on agreed strip | Strip pack `match3_quad_p10_31` + editable `/match3-m1` review; provisional auto-QA 39/39; **P_emit ≥ 0.80**, clear-ball **R ≥ 0.80** after human confirm | Acceptance |
| **F1** | Soft-dual fallback: agree cluster with combined conf &lt; 0.80 falls through to solo (do not silent-drop) | Unit + eng-loop: weak dual + strong out-of-cluster cam → solo emit | Recover strong solos blocked by soft pairs |
| **F2** | Solo = **max conf** among mapped cams, not weight-seed only | Unit + eng-loop: weak high-support + strong low-support disagree → emit strong | Stop ghosts blocking ≥0.80 cams |
| **F0** | Detect-tick **hold** (`fuse_balls_with_hold`, `HOLD_MAX_GAP=4`) — not Phase 2 RNN fusion | P8/P10 human strips: P≥0.90 and R≥0.80 at hold=4 (A/B vs hold=2) | Clear-R lift across silent ticks; no extra detect latency |
| **F3** | Ghost prune: drop weak maps far from max-conf anchor (`GHOST_CONF=0.45`) | Unit + eng-loop; multi-strip A/B keeps P_emit ≥ 0.80 | Stop P1/P7 ghosts vetoing strong cams |
| **M2** | Second strip `match3_quad_p8_87` (P8 @ 1:27) | Both strips P_emit ≥ 0.80; product clear_R (F0) ≥ 0.80 | Goals evidence beyond one clip |

Emit gate and agree radius stay fixed. Soft clear FNs with **no** cam ≥ 0.80 on a detect tick are a **model** problem after F0–F3.

**Clear-ball product-wide (R0+R1 front):** score packs with product F0; random proxy 0.525→0.625 is measurement-only. FN audit + dead ends: [MATCH3_CLEAR_BALL.md](MATCH3_CLEAR_BALL.md), [`reports/ball_testing/`](../../reports/ball_testing/).

## Product ratings (eng-loop)

| Rating | ≥9 when |
|--------|---------|
| **product_goals** | ≥2 strips; each P_emit ≥ 0.80; F0/carry clear_R ≥ 0.80; ≥1 strip carry_R ≥ 0.90; random gallery present |
| **product_post** | F0–F3 in `match3_xy.py`; A/B winner includes F0; random gallery emit ≥ 150; EMIT_CONF = 0.80 |

## Hard no

- Widen agree beyond 4 m or average distant cams
- Remove hull gate entirely without L2
- Replace emit 0.80 with lower solo thr
- FIFA 105×68 / invent penalty spots on Pitch 1
- Phase 2 RNN / occlusion-merge as a Phase 1 requirement

## Wire

- Thr: `scripts/gold_set/demo_locked_oos_pitchmap.py` + Match 3 gallery (not Match 2 `TOP_LEFT_THR_BY_CAM` globally)
- Calib: `scripts/gold_set/match3_landmarks.py`, marker UI
- Map/fuse: `src/mapping/match3_xy.py` (`fuse_balls`, `fuse_balls_with_hold`)
- Fuse A/B: `scripts/gold_set/ab_match3_fuse_post.py` → `reports/eval_match3/improve_eng_loop/f_post_ab.json`
- Pitch meters: `docs/product/PITCH1_DIMENSIONS.json`
- Eng-loop: `scripts/gold_set/eng_loop_match3_improve.py` (plan ≥ 9/10; includes `f_post`)
- M1 strip: [MATCH3_M1_STRIP.md](MATCH3_M1_STRIP.md) (`match3_quad_p10_31`)

## Checks

- Eng-loop plan score **≥ 9/10**
- Landmark round-trip ≤ 0.15 m after L1/L2
- Fuse: 2-cam ≤4 m → median; disagree → no midpoint; solo conf 0.4 → no emit; F1/F2 unit cases pass
- F post A/B winner ⊆ {F1, F2, F1+F2, F1+F2+F0} with P_emit ≥ 0.80
- Camera id = video title (P9-004 → P9)
- Final: **P_emit ≥ 0.80**, clear-ball **R ≥ 0.80**
