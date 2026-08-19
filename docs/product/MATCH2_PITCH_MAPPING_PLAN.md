# Match 2 — pitch mapping + off-pitch bounds (loop plan)

**Goal:** Pixel → pitch `(x, y)` for Match 2 cams, then reject ball dets whose pitch coords fall outside the field (spare balls / bins on the brown track).

**Complements:** Existing turf color gate (`src/perception/pitch_mask.py`). Mapping is the geometric second check.

Product pitch bounds (`in_pitch_bounds`) default to **Pitch 1 / Field 1** ([PITCH1_DIMENSIONS.md](PITCH1_DIMENSIONS.md): 53.90 × 34.84 m). Match 2 auto-H scripts still hardcode FIFA 105×68 — that is **not** the official field; do not copy those numbers into Pitch 1 / Match 3.

## Why now

- Locked multicam pick is live; sideline FPs still hurt largest-ball.
- Color gate helps; pitch-bounds is the clean geometric gate once H exists.
- Phase 1 already lists coordinate mapping; code exists, Match 2 calib does not.

## Minimize user work

1. Prefer **auto** homography from landmarks (no click-calib unless auto fails).
2. Start with **Cam4plus + Cam5plus** only (masters that win most frames).
3. Do not re-label gold for this step.
4. Keep color gate on; add pitch-bounds as AND filter when H is available.

## Loop steps

| Step | Do | Done when |
|---|---|---|
| **A** | Inventory: any Match 2 H JSON on disk? Document paths. | Note in this file |
| **B** | Auto-estimate H for Cam4plus + Cam5plus from Top Left / Bottom Right stills | `reports/eval_match2_v10/match2_pitch_calib/*.json` |
| **C** | Sanity: project pitch corners / center circle back to image; visual check | `*_overlay.jpg` + pass/fail note |
| **D** | Helper: `in_pitch_bounds(x, y, margin_m=…) → bool` | Unit tests |
| **E** | Wire into `pick_product` / detect when `frames_by_cam` + H available | Sideline ball rejected on Top Right OOS |
| **F** | Re-run Top Right pitchgate demo; compare none-rate vs color-only | Stats delta in survey note |

## Stop / handoff

- If auto H fails landmarks → stop loop; ask user for one manual calib click session (`scripts/calibrate_homography.py`).
- If H looks good → finish E–F without user.

## Auto-v2 (landmark improve loop)

| Step | Do | Done when |
|---|---|---|
| **A1** | `scripts/gold_set/auto_match2_homography.py` multi-frame CLI | Runnable for Cam4+/Cam5+ |
| **A2** | Hard overlay gate + `*_overlay_v2.jpg` | Rejects prior bad autos |
| **A3** | ≤3 retries (stems/frames/refine); detect @1280 then scale H | Status notes attempt |
| **A4–A5** | `in_pitch_bounds` + wire into pick | Only if **both** cams `pass=true` |
| **A6** | If exhausted → manual handoff; idle loop | User click session |

## Status

- **2026-08-16:** Plan created. Color gate shipped (`9642282`).
- **A done:** No prior Match 2 H JSON on disk.
- **B done (numeric):** Auto H written for Cam4plus + Cam5plus under `reports/eval_match2_v10/match2_pitch_calib/` (OpenCV 5 Hough `(N,4)` unpack fixed).
- **C FAIL (v1):** Overlays did **not** follow real pitch lines (Cam4 ≈ axis-aligned frame quad; Cam5 ≈ thin mid strip).
- **2026-08-17 Auto-v2:**
  - **A1–A3 done.** CLI + gate + 3 retries. Artifacts: `match2_pitch_calib/auto_v2_status.json`, overlays `*_overlay_v2.jpg`.
  - **Cam4plus: PASS** on `tl_dist_off` → `Cam4plus_auto_v2.json` (do not promote alone).
  - **Cam5plus: FAIL** all 3 attempts (`thin_axis_aligned_band` / thin mid strip) → `Cam5plus_auto_v2_FAILED.json`.
  - **A4–A5 blocked:** both cams must pass. Helper `src/mapping/pitch_bounds.py` exists but is **not** wired into `pick_product`.
  - **Manual fallback handoff:** please run `scripts/calibrate_homography.py` for **Cam5plus** (and optionally re-validate Cam4plus). Still frames under `reports/eval_match2_v10/match2_pitch_calib/stills/`.
- **2026-08-17 FOV-aware auto:** Added `src/mapping/fov_aware_homography.py` (FOV wedge prior + visible line junctions). Cam5 center circle is **clipped** by the frame edge, so auto still fails overlay sanity; Cam4 keeps prior `auto_v2` pass. Manual click order for Cam5 is now `visible_side` (touchline×halfway, touchline×goal, circle×halfway, center) — not full center-circle cardinals.
- Loop: every **15m** (`AGENT_LOOP_TICK_auto_h`) — idle until `Cam5plus_manual.json` lands; then resume D–F.
