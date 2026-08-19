# Match 3 ball (x, y) + combine

Use the eight **manual 4-click** homographies in `reports/eval_match3/match3_pitch_calib/`. Camera id = video title (`P9-004.mp4` → P9). Pitch meters are **Pitch 1 / Field 1** ([PITCH1_DIMENSIONS.md](PITCH1_DIMENSIONS.md): **53.90 × ~34.8 m**, not FIFA 105×68). Origin center, **+x north (P6)**, **+y left** from P1.

## Why the old path is wrong

`pick_product` chooses **one** camera by **largest pixel box**, then maps that box **center** through H or a **FOV wedge**. Largest-on-sensor is a close-up cam, not the best pitch (x, y). A 4-click H is only true **near the clicked landmarks**. Averaging two cameras that disagree by tens of meters invents a ghost ball.

## Map (per camera)

1. Load `{cam}_manual.json` only. No FOV fallback when H exists. No H → skip that cam (do nothing).
2. Scale pixels to calib `image_wh` (stills and detect are 1920×1080).
3. Ground point = **bbox foot** `(cx, y+h)`, not box center (ball sits on grass).
4. `H @ (x, y, 1)` → Pitch 1 / Field 1 meters.
5. Drop if off-pitch (`in_pitch_bounds`, margin 1 m) or `w ≈ 0`.
6. **Local support:** weight by distance to the convex hull of that cam’s clicked image points. Far from the hull → drop (H extrapolation).

## Combine (same timestamp)

1. Take each cam’s best remaining mapped ball (`conf × support`).
2. Seed = highest weight. Cluster = others within **4 m** on the pitch.
3. **Agree (≥2 cams):** emit **median** (x, y). Combined conf = `1 − Π(1 − c_i)`. Winner cam = highest support in the cluster (for the video pane).
4. **Disagree:** do not average. Emit the seed only if it alone clears the emit gate.
5. **Emit gate:** combined conf **≥ 0.80**. Below that → drop / do nothing.
6. One ball per frame. No temporal RNN / occlusion-merge (Phase 2).

Pixel largest-ball and green-turf gates stay as **detection** filters. They do not choose pitch (x, y).

## Wire

- `src/mapping/match3_xy.py` — load, map, fuse.
- Match 3 pitchmap gallery / dual-pane: fuse, then draw.
- `apps/batch_pipeline.py` — if the file is a Match 3 cam, set that cam’s H; ball uses bbox foot.
- Match 2 8-cam `pick_product` unchanged until Match 2 has the same 4-click maps.

## Checks

- Landmark round-trip: each calib’s image clicks map back to Pitch 1 points (~0 m, DLT).
- Two cams ~2 m apart → fused median, not the bigger pixel box.
- Two cams ~30 m apart → no midpoint.
- Off-pitch / low support / conf 0.4 singleton → no emit.
- Playwright/unit eng-loop **9+/10** on plan + map + fuse + emit + cam ids.
