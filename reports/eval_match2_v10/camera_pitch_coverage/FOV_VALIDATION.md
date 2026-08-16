# Camera FOV validation (loop)

## Method
1. Synced Top Left frames f150 + f097 from all 8 cams.
2. Map FOV polygons from visible landmarks (goals, center circle, touchlines).
3. Re-anchor ball-px model to detections: P10≈21, P7≈13, Cam4plus≈26–27.

## Screenshot vs old diagram
| Cam | Fix |
|---|---|
| P7 | Was left wedge → almost **full left half** |
| P10 | Wider top-left→center band |
| P1 | Not left-box only → **midfield** view |
| Cam4plus | Not full pitch → **left half / sideline master** |
| Cam5plus | Overlaps Cam4plus heavily |
| P8 | Right half mid→goal (confirmed) |
| P12 | Overlaps left-half with P7 |

## Optimize (conceptual)
1. **Pool:** Cam4plus + P-cams; don’t pretend one wide cam covers all.
2. **Split masters:** re-aim Cam5plus to **right** half to cut left overlap.
3. **Selection:** prefer larger-ball cam (Cam4plus/P10) over tiny far views (thr floors).
4. **Zoom/aim:** touchline cams for local ≳15–20px; hand off by region.
5. **Don’t add cams** until overlap is reduced — P7/P12/4+/5+ fight over the same left half.
