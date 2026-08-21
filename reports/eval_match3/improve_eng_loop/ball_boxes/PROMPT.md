# Eng-loop prompt: ball bounding boxes on Match 3 WHOLE PITCH mosaic

**Goal:** Orange RF-DETR ball boxes must sit **on the ball** in every mosaic tile (defish ON, locked layout). Score ≥ **9/10**.

## Locked product context

- Layout: Top **P10|P9 (180°)**, Bottom **P7|P8** (`match3_camera_layout`)
- Defish **ON** for P7–P10 (`match3_defish`); detect **after** defish on shown pixels
- Pitch 1 meters — not FIFA 105×68
- Precision-first: top-1 ball per tile (`keep_top1_ball`)

## Pass criteria (each scored 0–10, mean ≥ 9)

1. **Presence:** When a clear ball is visible in a tile, an orange box + `BALL` label is drawn.
2. **Lock:** Ball box center within **≤ 25 px** of the ball blob center on the **final displayed tile** (after defish, 180° rotate if any, letterbox).
3. **No ghost:** No orange ball box on empty grass far from any ball-like blob (> 80 px).
4. **Size:** Box covers the ball (IoU-ish): ball disk mostly inside box; box not > ~4× ball diameter.
5. **Layout intact:** Mosaic still P10|P9 / P7|P8 with correct 180° on top row.

## Method

1. Pick frames with a clear ball (e.g. 4102 and 1–2 more from ball-export nav).
2. Build mosaic with `apply_defish=True`, live RF-DETR + `keep_top1_ball`.
3. Per cam: screenshot tile crop; measure box-vs-ball offset (HSV orange ball and/or bright small blob near box).
4. If FAIL: fix draw/transform path only (letterbox scale, rotate, defish detect order) — do **not** turn defish off.
5. Re-score until ≥ 9/10; write `reports/eval_match3/improve_eng_loop/ball_boxes/score.json` + JPGs.

## Product draw rules (locked by this eng-loop)

- Detect **after** defish on shown pixels.
- Coach ball box: **min side ≥ 32 px**, centered on det (so it is a real orange rectangle, not a dot).
- Drop ball dets with **conf &lt; 0.30**.
- Drop ball dets that **fail `map_ball_box`** (off-pitch / low hull support) — precision-first.
- Drop balls that **overlap a player** or sit on the **frame edge**.
- **One ball for the whole mosaic:** after all cams detect, keep only the highest-conf mapped ball (`_keep_single_mosaic_ball`) so the orange box matches the single yellow ball on Pitch 1.
- Layout stays Top P10|P9 (180°) / Bottom P7|P8.

## Out of scope

- Player box polish, pitch fuse team colors, Streamlit connection, changing camera layout.
