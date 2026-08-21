# Match 3 defish — evidence and product wiring

Brown `cv2.undistort` on **P7–P10** before homography. Landmarks and H are fit on **defished** stills; raw detection feet are undistorted in `map_ball_box` before H. Do not map raw fisheye pixels through a defish H.

## Locked params (tags.json)

| Cam | k1 | k2 | α | Notes |
|-----|-----|-----|---|-------|
| P7 | −0.30 | −0.08 | 0.80 | |
| P8 | −0.30 | — | 0.80 | |
| P9 | −0.40 | — | 0.80 | |
| P10 | −0.32 | — | 0.80 | |

Dashboard: `scripts/gold_set/match3_fisheye_dashboard.py` → `/match3-fisheye`. Tags: `reports/eval_match3/fisheye_dashboard/tags.json`.

## Product path

- **Landmark stills:** `match3_landmarks.py` applies tags when extracting; calibs store `undistort` + fingerprint.
- **Map:** `src/mapping/match3_xy.map_ball_box` — undistorts raw feet when calib has `undistort` (default). Pass `apply_undistort=False` for A/B only.
- **Pitchmap demo:** `demo_locked_oos_pitchmap.py --no-undistort` for nodefish gallery.
- **Tests:** `scripts/gold_set/test_match3_xy.py` (`test_map_ball_uses_undistort_for_raw_foot`).
- **A/B harness:** `scripts/gold_set/ab_match3_undistort_map.py` → `reports/eval_match3/improve_eng_loop/undistort_map_ab_random5.json`.
- **C1 FN audit:** `scripts/gold_set/fn_audit_match3_quad.py` → `reports/eval_match3/improve_eng_loop/c1_fn_audit.json`.

Detection still runs on **raw** frames; only the pixel→pitch step is defished.

## vs PoC goals

From [MATCH3_MULTICAM_IMPROVE_PLAN.md](MATCH3_MULTICAM_IMPROVE_PLAN.md):

| Role | Metric | Target |
|------|--------|--------|
| **Precision** | `P_emit` — emitted fuses within 4 m of gold | **≥ 0.80** |
| **Recall** | `clear_ball_R` — emit rate on clear-ball frames | **≥ 0.80** |

Emit gate stays **conf ≥ 0.80**.

### Before vs after (same v12 dets, F0 hold)

**Before (nodefish):** defish H, **no** undistort on detection feet (`apply_undistort=False`).  
**After (product defish):** undistort P7–P10 feet, then H.

#### Random 5 pitchmap gallery (750 frames, 5×150)

Manifests: `pitchmap_gallery_defish/` vs `pitchmap_gallery_nodefish/`.

| Metric | Nodefish | Defish | Δ |
|--------|----------|--------|---|
| Emits | 189 | **247** | +31% |
| Multi-cam agree | 3 | **54** | +51 |
| P7–P10 maps (A/B totals) | 98 | **224** | +129% |

Largest clip (**15:31**, `rand_931`): emit **22 → 76**, agree **1 → 49**.

Viewer (after `python3 serve_viewer.py`):

- Defish: `/match3-pitchmap-defish`
- Nodefish: `/match3-pitchmap-nodefish`

#### Undistort map A/B (1495 frames, 5 random det caches)

`reports/eval_match3/improve_eng_loop/undistort_map_ab_random5.json`:

| Metric | Raw (no undistort) | Product undistort |
|--------|-------------------:|------------------:|
| Emits | 262 | **346** |
| Multi-cam agree | 1 | **34** |
| Clear-ball proxy R | **0.30** (21/71) | **0.63** (45/71) |
| P7–P10 maps | 98 | **224** |

Clear-ball proxy R = `clear_emit / clear_frames` on unlabeled random packs — directional only, not gold `P_emit`.

#### Labeled M1 strips (gold)

`reports/eval_match3/improve_eng_loop/m1_provisional.json` — scored with **product defish** wired:

| Strip | P_emit | clear_ball_R | PoC |
|-------|--------|--------------|-----|
| `match3_quad_p10_31` | **1.00** | **0.81** | pass both |
| `match3_quad_p8_87` | n/a (≈no emits) | **0.02** | fail recall |

No clean gold before/after for nodefish — labels and H were built under defish landmarks. Geometry: skipping undistort on edge landmarks maps **~10× worse** (unit test `test_map_ball_uses_undistort_for_raw_foot`); P7–P10 emits without defish would break precision.

## Bottom line

- **Defish helps a lot** on recall, multi-cam agree, and P7–P10 map count; required for correct fisheye geometry.
- **Does not alone** hit **0.80** clear-ball R on random unlabeled packs (~0.63 proxy) or fix P8 strip recall.
- **Do not remove** defish from the product map path or revert P7–P10 landmarks to raw stills.
- **Do not** tune stronger k1 expecting recall fixes — remaining gap is quad coverage, landmarks, dets, fuse (see improve plan).

## Related

- Capture + fisheye tags: [MATCH3_CAPTURE.md](MATCH3_CAPTURE.md)
- Improve loop: [MATCH3_MULTICAM_IMPROVE_PLAN.md](MATCH3_MULTICAM_IMPROVE_PLAN.md)
- Alpha tuning runbook: [../runbooks/FISHEYE_ALPHA_GUIDE.md](../runbooks/FISHEYE_ALPHA_GUIDE.md)
