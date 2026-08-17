# Match 2 — pitch mapping + off-pitch bounds (loop plan)

**Goal:** Pixel → pitch `(x, y)` for Match 2 cams, then reject ball dets whose pitch coords fall outside the field (spare balls / bins on the brown track).

**Complements:** Existing turf color gate (`src/perception/pitch_mask.py`). Mapping is the geometric second check.

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

## Status

- **2026-08-16:** Plan created. Color gate shipped (`9642282`).
- **A done:** No prior Match 2 H JSON on disk.
- **B done (numeric):** Auto H written for Cam4plus + Cam5plus under `reports/eval_match2_v10/match2_pitch_calib/` (OpenCV 5 Hough `(N,4)` unpack fixed).
- **C FAIL:** Overlays do **not** follow real pitch lines (Cam4 ≈ axis-aligned frame quad; Cam5 ≈ thin mid strip). Auto landmarks are not trustworthy for bounds.
- **Stop / handoff:** Need one **manual** click calib per master cam (`scripts/calibrate_homography.py` or equivalent) for Cam4plus + Cam5plus. Do not wire auto H into `pick_product`.
- Loop: every **15m** (`AGENT_LOOP_TICK_pitch_map`) — on next ticks, only advance D–F after manual H lands; otherwise idle.
