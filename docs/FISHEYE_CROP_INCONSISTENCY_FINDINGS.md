# Why fisheye “Fixed” frames are still inconsistent

## What we see

- **Frame 0**: sometimes shows visible black (curved borders) in the “Fixed” column.
- **Frame 363** (and others): sometimes looks clean.
- Same pipeline and code for every frame, but the result varies.

## Root cause (why it’s inconsistent)

### 1. **95% non-black is too loose**

In `crop_black_borders()` we keep a row/column if “≥ 95% of its pixels are non-black”. So we allow **up to 5% black per row/column**. That still leaves visible black:

- 5% of 1440 px ≈ 72 pixels → a clear line or band.
- The **final square** can still contain rows/cols with 1–3% black (measured: Frame 0 had 23 rows and 27 cols with >1% black; Frame 363 had 50 rows and 78 cols).

So the crop does not remove all visible black; how much remains depends on the frame.

### 2. **Crop bounds depend on the frame**

Because black in the remapped image varies by frame (content, lighting, lens):

- **Frame 0**: bottom rows have more black → we crop to `y2 = 1915` (drop 5 rows).
- **Frame 363**: bottom rows are cleaner → we keep full height `y2 = 1920` (crop 0 rows).

So `(y1, y2, x1, x2)` are **different per frame**. Then we center-crop that rectangle to a square. So:

- The square for Frame 0 is the center of a 1915×1440 crop.
- The square for Frame 363 is the center of a 1920×1440 crop.

Different crop boxes + different black distribution → some squares end up with more visible black (e.g. Frame 0), others with less (e.g. Frame 363). That’s the inconsistency.

### 3. **x1/x2 use the full image height**

We compute `x1`, `x2` using **full-column** stats: `is_black[:, x]` over all rows. So we’re not forcing “no black in the band we actually keep (y1:y2)”. If black is curved:

- A column can be “95% non-black” over the full height (because the middle is content) but still have black in the band `y1:y2`.
- So the rectangle `[y1:y2, x1:x2]` can still include columns that have visible black in that band.

That can make the visible black in the final crop/square a bit worse or more variable.

## Summary

| Cause | Effect |
|-------|--------|
| 95% non-black threshold | We keep rows/cols with up to 5% black → visible bands in the square. |
| Frame-dependent crop (y1,y2,x1,x2) | Different frames get different crop boxes → different amount of black in the center square. |
| x1/x2 from full columns | Horizontal crop can include columns that are black in the kept band (y1:y2). |

So “some fixed, some not” comes from: **per-frame crop bounds** plus **a threshold that still allows visible black** in the final square.

## Recommended fixes

1. **Stricter threshold**  
   Use **99%** (or 99.5%) non-black instead of 95% so we trim until there’s almost no black left. Then the square will be consistently clean.

2. **Compute x1/x2 from the band**  
   After computing `y1`, `y2`, compute `x1`, `x2` from **only** `is_black[y1:y2, :]` so the horizontal crop matches the vertical band and curved black doesn’t slip in.

3. **Optional: fixed square from center**  
   If you always want the same field of view, you could define the square from the **center of the remapped image** with a fixed size (e.g. 90% of the shorter side) instead of “crop black then center-crop to square”. Then every frame would use the same region; you’d still want to trim obvious black first (e.g. with the stricter threshold above).

Implementing (1) and (2) should make the “Fixed” column consistent and remove the remaining black.
