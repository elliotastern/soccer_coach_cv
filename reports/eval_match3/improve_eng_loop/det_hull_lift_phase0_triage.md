# Detector + hull lift — Phase 0 triage (2026-09-02)

Fresh audits with current calibs/weights. Product fuse unchanged (F1+F2+F0+F3, EMIT_CONF=0.80, AGREE_M=4, MIN_SUPPORT=0.20).

## Sources

| Script | Output |
|--------|--------|
| `fn_audit_match3_quad.py` | `c1_fn_audit.json` |
| `fn_audit_match3_random.py` (holdout) | `r1_random_fn_audit_holdout.json` |
| `fn_audit_match3_random.py` (tune gallery) | `r1_random_fn_audit_tune.json` |
| `holdout_other_cam_funnel.py` | `holdout_other_cam_funnel.json` |

## Fuse vs map/det

Holdout clear FNs with another cam mapped ≥0.80: **0/26** (`frac=0.0`). Not a fuse-wiring gap.

## Bucket counts (clear FNs)

| Pack | mapped_conf_below_emit | focus_map_fail / low_support | no_det_focus |
|------|------------------------:|-----------------------------:|-------------:|
| Holdout gallery | 8 | 18 (all `P_Goal1\|low_support`, **1 cache**) | 0 |
| Tune gallery | 8 | 7 (clip-specific) | 0 |
| Strip P10 | 3 | 0 | 0 |
| Strip P8 | 10 | 0 | 28 |

Holdout proxy clear_R ≈ **0.892** (215/241).

## Systemic vs clip (`≥2` caches)

**Systemic (detector conf):**

- Holdout: `mapped_conf_below_emit|P10`, `mapped_conf_below_emit|P6`
- Tune: `mapped_conf_below_emit|P8`

**Clip-only (do not hull):**

- Holdout: `focus_map_fail|P_Goal1|low_support` (18 FNs, one cache `t01767.3s`)
- Tune: `focus_map_fail|P7|low_support` (one cache)

## Status 2026-09-03 (post-v13 / v14 in flight)

See `det_hull_lift_phase0_triage_v13.md` for full post-promote triage.

- Holdout rebuilt: proxy R **0.893** (gate held).
- Next specialty: **v14** residual pack built (709 train); Mac MPS train **265→275** running (`/tmp/v14_residual_mac_train.log`).
- Catch still unreachable (Tailscale timeout) — @1288 deferred.

## Step 3 decision (locked)

Skipped. Clip-only `focus_map_fail|…|low_support` must not drive hull expands.

## Landmarks / L2 (locked closed)

User confirmed (2026-09-02): **no remaining unlabeled landmarks in camera FOVs**. Do not chase L2 / extra clicks. Next levers are detector (Catch @1288 preferred) only — not fuse hyperparams, not inventing mid-pitch marks.
