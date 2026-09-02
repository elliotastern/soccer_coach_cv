# Ball testing — dead ends (do not re-run as “new”)

Living ledger. After any failed A/B, add one row. Numbers live under `reports/eval_match3/improve_eng_loop/`; this file is the human memory.

| Tried | Metric / pack | Result | Why not again |
|-------|---------------|--------|----------------|
| Stronger fisheye k1 alone | Random / strips | No product-wide R fix | Geometry already locked; see MATCH3_DEFISH |
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

## Worked (keep)

| Tried | Result |
|-------|--------|
| Spot-check holdout P10 `low_support` | 73 in-FOV feet → proceed hull |
| P10 `hull_image_points` (lower FOV) | Holdout R **0.556 → 0.867**; strip P_emit held |
| H1 MIN_SUPPORT **0.25 → 0.20** | Holdout R **0.867 → 0.884**; strip P held → **promoted** |
| Other-cam ≥0.80 on clear FNs | **0/28** — residual is map/det, not fuse drop |
| L1 white-line re-click P8/P9 (≤18 px; hull preserved) | RT ≤ 0.15; P10 clear_R **0.876→0.879**; P8/holdout flat; strip P held |
| **3D + smart hybrid** (`pick_3d_hybrid`: 3D when agree, else F0–F3) | Holdout clear_R **0.892** (parity); agree **0.343** (parity); P10 strip held; **promote_3d=true** | Opt-in via `fuse.mode: triangulate_3d`; default stays `pitch_merge` · `fuse3d_ab.json` |

## Residual (not fixed)

- Clip-specific `focus_map_fail` (e.g. P_Goal1 low_support on t1767) — do **not** hull for one seed.
- Systemic `mapped_conf_below_emit` on **P6** (2 holdout caches, small n) — only if holdout regresses later.

See [CLEAR_BALL_FRONT.md](CLEAR_BALL_FRONT.md) · [HOLDOUT_BASELINE.md](HOLDOUT_BASELINE.md).

- H-consistency baseline: holdout pairwise map span median **~13 m** → improve landmarks/auto-H before shrinking agree (`h_consistency/`).
