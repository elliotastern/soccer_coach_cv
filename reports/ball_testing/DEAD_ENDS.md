# Ball testing — dead ends (do not re-run as “new”)

Living ledger. After any failed A/B, add one row. Numbers live under `reports/eval_match3/improve_eng_loop/`; this file is the human memory.

| Tried | Metric / pack | Result | Why not again |
|-------|---------------|--------|----------------|
| Stronger fisheye k1 alone | Random / strips | No product-wide R fix | Geometry already locked; see MATCH3_DEFISH |
| **Manual tags retune** (P7 k1 −0.30→−0.41, P10 −0.32→−0.49; P8/P9 unchanged) | Strips P_emit/clear_R flat; **P10 mean err 0.88→1.21 m**; holdout proxy R **0.892** flat, emit 554→552 | Visual k-only vs locked H | Do not promote into `*_manual.json` without landmark re-click · `manual_defish_tags_ab.json` |
| Defish (D0) alone | Random clear proxy | ~0.30 → ~0.63; still &lt; 0.80 | Necessary, not sufficient |
| Quad / P6 hull expands (C1/C3) for “product-wide” | Quad ↑ (~0.84); random stuck ~0.53 | Seed/FOV overfit when chasing tune pack | Prefer **holdout** gate + systemic FOV only |
| Proxy score without F0 hold | Random clear proxy | **0.525** undercount | Measurement bug — product already holds |
| R0 wire F0 into `score_cache` | Same packs | **0.625** | Measurement only — **no product change** |
| Holdout pack built (seed 20260821) | New times; freeze R **0.556** | Honest gate set | Do not retune on these starts blindly |
| Lower EMIT_CONF / widen AGREE_M / drop hull | — | Forbidden | Phase 1 precision-first Hard no |
| Phase-2 RNN / occlusion merge for Phase 1 clear-R | — | Out of scope | product_phase_scope |
| t552 one-clip det (map-first residual) | Holdout after P10 hull | **Skipped** — map FNs 22 vs conf 6 | Conf not majority; no MPS burn |
| **Defish-detect A/B** (undistort frame → RF-DETR → map no foot undistort) | Holdout proxy **0.892→0.936**; quad **0.931→0.876**; P10 strip clear_R **0.879→0.817**; P8 **0.846→0.820** | Holdout proxy ↑ but **labeled strips + quad down** | Keep batch **raw detect + foot undistort**; mosaic-only path OK · `defish_detect_ab.json` |

| **F4 reprojection prune** | holdout/strips A/B | **Skipped** — gates failed | See `reports/eval_match3/improve_eng_loop/f4_reproj_ab.json` |
| **Player F4 reprojection** (64 px, 3 frames) | player_map_funnel frames | **Skipped** — 13→1, 10→1 on fr 2400/3600 | Default off; `player_reproj_ab.json` |
| **3D triangulate fuse** (pure, no fallback) | holdout clear_R **0.801**; agree **0.411**; P10 strip clear_R **0.732** | **Skipped** — below 0.884 holdout gate | Default off · `fuse3d_ab.json` |
| **3D + F0–F3 hybrid** (naive: 3D solo before 2D) | holdout clear_R **0.892**; agree **0.231** | **Skipped** — agree below baseline 0.343 | Replaced by smart `pick_3d_hybrid` · `fuse3d_ab.json` |
| **3D + UKF** | holdout clear_R **0.809**; agree **0.473** | **Skipped** — clear_R below gate | Default off · agree lift only · `fuse3d_ab.json` |
| **Ball foot-point modes** (inset/center/radius) | holdout/strips A/B | **Skipped** — no gate-safe win vs bottom | See `reports/eval_match3/improve_eng_loop/ball_foot_ab.json` |
| **Soft / gated F4** (agree-gate + n≥3 prune) | holdout/strips A/B | **Skipped** — no gate-safe win vs F0–F3 | See `reports/eval_match3/improve_eng_loop/soft_f4_ab.json` |
| **ghost_conf 0.45→0.80** | holdout/P10 A/B | **Skipped** — no gate-safe win | See `reports/eval_match3/improve_eng_loop/ghost_conf_ab.json` |
| **AGREE_M shrink** 4→3.0/2.5 | holdout agree **0.343→0.337/0.321**; proxy held | **Skipped** — agree fell; product stays 4.0 | `agree_m_ab.json` |
| **H-first 3D+UKF re-A/B** (same calibs) | hybrid parity clear_R **0.892** agree **0.343**; UKF clear_R **0.809** | **No product change** — still parity / UKF kills recall | `fuse3d_ab.json` + `h_consistency/` |
| **v13 residual 1-epoch only** (Mac MPS @560) | P10 clear_R **0.879→0.866**; P8 **0.846→0.860**; P_emit **1.0** | **No promote** — too short / P10 regress | Superseded by 10-epoch promote |

## Worked (keep)

| Tried | Result |
|-------|--------|
| Spot-check holdout P10 `low_support` | 73 in-FOV feet → proceed hull |
| P10 `hull_image_points` (lower FOV) | Holdout R **0.556 → 0.867**; strip P_emit held |
| H1 MIN_SUPPORT **0.25 → 0.20** | Holdout R **0.867 → 0.884**; strip P held → **promoted** |
| Other-cam ≥0.80 on clear FNs | **0/28** — residual is map/det, not fuse drop |
| L1 white-line re-click P8/P9 (≤18 px; hull preserved) | RT ≤ 0.15; P10 clear_R **0.876→0.879**; P8/holdout flat; strip P held |
| **3D + smart hybrid** (`pick_3d_hybrid`: 3D when agree, else F0–F3) | Holdout clear_R **0.892** (parity); agree **0.343** (parity); P10 strip held; **promote_3d=true** | Opt-in via `fuse.mode: triangulate_3d`; default stays `pitch_merge` · `fuse3d_ab.json` |
| **v13 residual 10-epoch** (Mac MPS @560; Catch down; no new landmarks) | P10 clear_R **0.879→0.888**; P8 **0.846→0.904**; P_emit ≥**0.99** | **Promoted** product `ball_checkpoint` → `v13_residual_snaps` · `ab_v13_residual_vs_v12.json` |
| **v14 residual 10-epoch** (Mac MPS @560; no holdout train) | P10 clear_R **0.888→0.929**; P8 **0.904→0.908**; holdout proxy **0.96**; P_emit ≥**0.97** | **Promoted** product `ball_checkpoint` → `v14_residual_snaps` · `ab_v14_residual_vs_v13.json` |
| **Constrained P1/P6 landmark nudge** (existing IDs; min-disp; kill-switched) | RT **1.12/2.43→0.13/0.0**; eng_loop `l2_overlap` **6→10**; strips/holdout held | **Promoted** · backups `*_pre_nudge_20260903_115944.json` · `overnight_top1f_nudge_result.json` |
| **Color-gated soft merge** (live 2.2 hard + same-team soft to 3.2) | Hold consensus **6.68→7.01** (+0.33); full kit consensus **7.1→7.35**; composite **9.01→9.1** | **Promoted** then superseded by mutual-nearest · `ab_color_gated_merge_kit_holdout.json` |
| **Mutual-nearest soft extend** (same-team cross-cam solos to 4.5 m) | Hold **7.01→7.42** (+0.41); full consensus **7.35→7.73**; composite **9.1→9.21** | **Promoted** · `PLAYER_MERGE_SOFT_M_LIVE=4.5` · `ab_mutual_nearest_soft_extend.json` |
| **Match3 full-cam kit** (add P1/P6 vs quad-only; tune-fit freeze) | Hold consensus **6.54→9.11**; mfc **0.11→0.51**; composite **9.57** | **Path confirmed** — Match4 needs Catch for P1/P6 · `ab_match3_fullcam_kit_freeze.json` |
| **Match3 Goal1/Goal2 in kit** (full8 vs full6 freeze wB) | Cons **9.31→9.53**; composite **8.95→8.96** (&lt;+0.2 bar) | **Optional / no promote** · `ab_match3_full8_goals_kit_freeze_wB.json` |
| **Kit centroid polarity select** (tune mean_blue band → freeze) | Tune blue in-band; hold B still mean_blue **0.26** / composite **8.95** (real share, not flip) | **Dead end** · `ab_match3_polarity_select_kit.json` |
| **Constrained P7–P10 quad nudge** (for kit mfc) | Kit full consensus **7.1→6.75**; mfc **0.22→0.14** at merge 2.2 | **Restored** · `ab_kit_after_quad_nudge.json` · `quad_nudge_restored.json` |

| **Widen PLAYER_MERGE_M_LIVE 2.2→3.2** (Match4 tune/hold) | Hold consensus **6.68→6.83** (<+0.3); tune looked better | **No promote** — H-span ceiling · `ab_player_merge_m_kit_holdout.json` |
| **Auto-H seed/cold + white-line snap/drop** on P1/P6 | Seed matched &lt;4; cold RT≈0 but only 4 mislabeled names (circular); snap ≈no lift; drop-4 still RT **0.20/0.67** &gt;0.15 | **No promote** — need human re-click of existing IDs · `h_consistency/snap_drop_h_ab.json` · `pitch1_auto_h.json` |

## Residual (not fixed)

- Clip-specific `focus_map_fail` (e.g. P_Goal1 low_support on t1767) — do **not** hull for one seed.
- Systemic holdout `mapped_conf_below_emit` on **P10/P6** (soft conf 0.50–0.79 majority) — detector specialty next; **do not train on holdout gallery**.
- Strip leftovers under v13: P8 soft/no_det (~23 label FNs); P10 essentially done (1 FN).
- Extra landmarks / L2 — **closed** (user: no unlabeled marks left in FOV). Do not invent mid-pitch points.
- Holdout gallery rebuilt under v13 (`det_cache_v13/`): proxy R **0.892→0.893**, gate held · `ab_v13_holdout_score.json`.

| **Crop-zoom redetect** on soft holdout FNs (pad×2–8 → DETECT size) | 0/6 crossed emit 0.80; often **lower** conf than fullframe | **Dead end** — soft balls are not a resolution crop problem |
| **F0b soft-hold renew** (conf≥0.55, support≥0.50, ≤4 m) | Holdout clear **0.978→0.993**; P10/P8 P_emit flat; clear_R +0.004 | **Promoted** · `ab_soft_hold_renew.json` · 2 holdout clear FNs left (soft 0.42/0.51) |
| **F3c SOLO_MIN_SUPPORT=0.50 all cams** | P10 P_emit **0.956→1.0** (fp 11→0); also blocked real P10 edge (support~0.28) | Superseded by P7-only scope |
| **F3c P7-only SOLO_MIN_SUPPORT=0.50** | P10 clear_R **0.914→0.953**, P_emit **1.0**; P8 flat; holdout **0.993** | **Promoted** · `ab_solo_min_support_scope.json` |

| **SAHI recover on strip no_det clear FNs** (v14, slice 640) | 2/24 rescues (≥0.80 near gold); most best conf 0.1–0.79 | **No promote** as product default — sparse; prefer v16 specialty · optional A/B later |
| **v16 fuse-silent residual** (resume v14 @285→300; strip FN pack) | P10 clear_R **0.953→0.966**; P8 **0.952→0.971**; P_emit **1.0**; holdout **0.993→0.987** (≪gate, within 0.01) | **Promoted** product `ball_checkpoint` → `v16_residual_snaps` · `ab_v16_residual_vs_v14.json` |
| **HOLD_MAX_GAP 8→24** (under v16+F3c) | P10/P8 clear_R → **1.0 / 1.0**; P_emit **1.0**; holdout **0.987** flat | **Promoted** · `ab_hold_max_gap_v16.json` |
| **t087 mosaic 15s re-render (v16 stack)** | Pitch ball_frac **0.61→0.48** vs prior v14 mosaic | Root cause: live `fuse_ball_product` static-ghost on **all** cams; strip M1 score had no ghost · `t087_ball_frac_autopsy.json` |
| **Static ghost all-cam (pre-scope)** | Strip clear_R **P10 1.0→0.595**, **P8 1.0→0.173** | Explains mosaic recall hole |
| **Static ghost P7-only** (`GHOST_STRICT_CAMS`) | Strips **1.0/1.0**; holdout **0.987** flat; t087 ball_frac **0.48→1.0** | **Promoted** · same pattern as F3c · `ab_static_ghost_product.json` |
| **F3c solo cascade** (skip blocked P7, try next ≥0.80) | Strips/holdout held **1.0 / 0.987** | **Promoted** · P7 ballcap no longer shadows real solos · leftover holdout FNs still soft focus (<0.80) · `ab_solo_cascade_ghost_p7_score.json` |
| **M1 score = live fuse_ball_product** | Strip/holdout parity with mosaic | **Promoted** · `score_match3_ball_m1.product_fuse_frame` |
| **v17 cross-cam soft residual** (resume v16 @300→315) | Strips P_emit/clear_R **1.0/1.0** (flat); holdout **0.987→0.984**; detect_ticks P10 **0.912→0.929** | **No promote** — gates held, no product clear_R lift · keep v16 · `ab_v17_residual_vs_v16.json` |

See [CLEAR_BALL_FRONT.md](CLEAR_BALL_FRONT.md) · [HOLDOUT_BASELINE.md](HOLDOUT_BASELINE.md).

- H-consistency baseline: holdout pairwise map span median **~13 m** — fuse hyperparams / inventing landmarks will not fix; detector + existing H only.

## v18 residual (tune-soft + strip focus) — no promote (2026-09-04)

Resume v16@300→315 with ~17 tune-gallery soft FNs (never holdout) + strip soft keep.
Strips stayed at P_emit/clear_R **1.0/1.0**; holdout proxy **0.987** (flat vs v16).
Same dead-end class as v15/v17: thin Match-3 soft residual does not lift remaining soft-focus FNs over emit 0.80.
Evidence: `reports/eval_match3/improve_eng_loop/ab_v18_residual_vs_v16.json`.

## Soft harvest → v19 residual — no promote (2026-09-04)

Non-holdout soft windows + pitchmap/quad caches → `soft_fn_harvest_plan_v19.json` (**n=30** soft FNs). Catch train: v18 pack + **+30 tune_soft** (replaced 17), resume v16@300→314.

| Pack | v16 | v19 |
|------|-----|-----|
| P10 P_emit / clear_R | 1.0 / 1.0 | 1.0 / 1.0 |
| P8 P_emit / clear_R | **1.0** / 1.0 | **0.979** / 1.0 |
| Holdout proxy R | 0.987 | 0.990 |

**Kill:** P8 P_emit regression. Same thin-soft dead-end class as v15/v17/v18. Keep **v16**. Evidence: `ab_v19_residual_vs_v16.json`. Next detector push needs thicker soft data (Match 4 / more Match 3), not another 15-epoch residual.

## Match 4 soft harvest → v20 residual — no promote (2026-09-04)

Dense Match 4 windows (40×5s @40s step) → **56** soft clear FNs (`soft_fn_harvest_plan_m4_only.json`). Resume v16@300→314 on v18 pack + Match4 tune_soft replace.

| Pack | v16 | v20 |
|------|-----|-----|
| P10 P_emit / clear_R | 1.0 / 1.0 | 1.0 / 1.0 |
| P8 P_emit / clear_R | 1.0 / 1.0 | 1.0 / 1.0 |
| Holdout proxy R | **0.987** | **0.980** |

**Kill:** holdout slight regress (no clear_R lift). Better than v19 (strips held) but Match-4 soft specialty still does not beat v16 on Match-3 holdout. Keep **v16**. Evidence: `ab_v20_residual_vs_v16.json`.


## Match3+Match4 soft → v21 residual — no promote (2026-09-04)

Combined soft pack **n=86** (M3 galleries 30 + Match4 harvest 56) into v18 base; resume v16@300→314.

| Pack | v16 | v21 |
|------|-----|-----|
| P10 / P8 P_emit · clear_R | 1.0 / 1.0 | 1.0 / 1.0 |
| Holdout proxy R | 0.987 | 0.984 |

**Gates held**, no clear_R lift → **no promote**. Keep **v16**. Better than v19/v20 (no emit/holdout kill). Evidence: `ab_v21_residual_vs_v16.json`, `soft_fn_harvest_plan_partial_combined.json`. Dense Match3 detect still filling caches for a thicker v22 later.

## Blur specialty v22 residual — no promote (2026-09-04)

Bounded try after blur autopsy (`blur_autopsy_soft_fns.json`: streak 30% / soft_round 57% → `pursue_blur_v22`). Resume v16@300→315 on streak/soft_round/tiny ×3 + soft residual (~3458 train) + v21 valid/anchors. Catch train `v22blur`; A/B `v22ab`.

| Pack | v16 | v22 blur |
|------|-----|----------|
| P10 / P8 P_emit · clear_R | 1.0 / 1.0 | 1.0 / 1.0 |
| Holdout proxy R | **0.987** | **0.983** |

**Gates held**, no clear_R lift (holdout −0.004) → **no promote**. Keep **v16**. Freeze thin soft/blur residual arc; do not start v23. Evidence: `ab_v22_blur_residual_vs_v16.json`.

## P1 near-touch hull expand — promoted (2026-09-04)

Kickoff P1 dets at conf≥0.80 mapped in-pitch but `low_support` (feet y~950–1040 after 4K→1080 scale; landmark hull max y~528). Same class as C3 P6.
**Fix:** `hull_image_points` on `P1_manual.json` (H unchanged). Strips/holdout held **1.0 / 0.987** (`ab_p1_hull_expand.json`).

## Static-ghost high-support P7 fade — fixed (2026-09-04)

Kickoff best-ball miss autopsy (`kickoff_miss_autopsy.json`): many 0.90 gaps were **real high-support P7 solos** faded/locked by static-ghost (ball sitting ~3 m for ≥12 source frames), plus ghosted prev blocking hold/soft-renew. P1 often `low_support`/P6 `off_pitch` (map, not det).
**Fix (promoted):** fade/lock only P7 solos with hull support &lt; 0.50 (ballcap signature); pass `support` on solo/hold emits; clear prev when gate rejects a ghosted hold. Strips/holdout flat **1.0 / 0.987** (`ab_ghost_support_gate.json`).

## Human blur gold ckpt sweep (2026-09-04) — report only

`ab_match3_human_blur_gold_ckpts.json`: v12/v14–v22 on 339 human boxes. Blurry (lap) subset R@0.30 ≈ **1.0** for v16+. Streaky R@0.30 stuck at **0.722** for all versions. No promote; product stays v16. See `docs/product/MATCH3_HUMAN_BLUR_GOLD.md`.
