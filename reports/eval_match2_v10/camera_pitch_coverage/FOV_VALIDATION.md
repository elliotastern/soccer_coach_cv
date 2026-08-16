# Camera FOV validation (loop)

## Method
1. Synced Top Left frames f150 + f097 from all 8 cams.
2. Map FOV polygons from visible landmarks (goals, center circle, touchlines).
3. Re-anchor ball-px model to detections: P10≈21, P7≈13, Cam4plus≈26–27.

## Screenshot vs old diagram
| Cam | Fix |
|---|---|
| P7 | Was left wedge → left half → midfield, but **near-left corner cut off** (photo: bottom frame ends before corner) |
| P10 | Wider top-left→center band |
| P1 | Not left-box only → **midfield** view |
| Cam4plus | Sideline master for **its** long side (not full pitch) |
| Cam5plus | **Opposite long side** from Cam4plus (user-confirmed) — not a Cam4 twin. Top Left stills can look similar in image space (fence-left framing) even when mounts differ |
| P8 | Right half mid→goal (confirmed) |
| P12 | Overlaps left-half with P7 |

## Optimize (conceptual)
1. **Pool:** Cam4plus + Cam5plus + P-cams; treat 4+/5+ as opposite-side masters.
2. **Do not** re-aim Cam5 onto Cam4’s half — they already split sides.
3. **Selection:** prefer larger-ball cam (often Cam4+/P10 on Top Left) over tiny far views (thr floors).
4. **Zoom/aim:** P-cam specialists for local ≳15–20px; hand off by region (P7 near-corner gap, P8 goal buffer).
5. **P-cam overlap** to cut: P7/P12 (and optionally P1) on the same left band — not Cam4 vs Cam5.
