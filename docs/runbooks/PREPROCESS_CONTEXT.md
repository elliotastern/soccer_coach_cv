# Preprocessing Pipeline: Context for Cursor / AI

**Use this in Cursor:** Open Chat (`Cmd+L`) or Composer (`Cmd+I`) and paste the **Context Prompt** below. It explains *why* the code does "weird" things (negative k, green masking, net mask) so the AI can help debug edge cases (e.g. net in foreground).

---

## Context Prompt (paste into Cursor)

```
I am working on a computer vision project to map soccer players from broadcast footage to a 2D top-down map. The current footage has two major issues: severe fisheye lens distortion and bright floodlights that break standard thresholding.
I need to implement a pre-processing pipeline in Python using OpenCV.
**The Goal:**
1. **Blind Defishing:** Undistort the video frames using a simple radial distortion model (estimating a negative `k` value) to straighten the curved touchlines. I do not have camera calibration parameters, so we must estimate the matrix on the fly.
2. **Green-First Filtering:** Eliminate the floodlights and sky by creating a "Field Mask" first. We will only look for "White" pixels (lines) if they are located inside "Green" pixels (turf).

**The Constraints:**
* Use `cv2` and `numpy`.
* The defishing function should allow me to tune the `k` value manually because the distortion amount is unknown.
* The color filtering should use HSV.
* Ignore the bottom 10% of the frame if possible, as there is a net in the foreground that might be detected as white lines.

Below is the starter code I have. Please review it, ensure it handles the video loop correctly, and explain where I should tweak the values if the lines are still curved.
```

---

## Script: `scripts/preprocess.py`

- **Defish:** `defish_frame(frame, k)` — radial undistort with tunable `k` (negative = fisheye correction).
- **Lines:** `isolate_lines(frame)` — green (HSV) field mask, then white (HSV) inside green; bottom `NET_MASK_HEIGHT` (default 15%) masked (net).
- **Live tuning:** Run from project root; `=` / `-` adjust k; `q` quit.

---

## How to Work the Script

1. **Run the script** (from project root):
   ```bash
   python scripts/preprocess.py
   ```
   Or with a video path:
   ```bash
   python scripts/preprocess.py data/raw/37CAE053-841F-4851-956E-CBF17A51C506.mp4
   ```

2. Look at the window **"1. Defished Reality"**.

3. Look at the far left or far right touchline. **Is it curved?**
   - If it curves like `(` (bowed out), press **`=`** to increase the negative k (straighten more).
   - If it curves like `)` (bowed in / pincushion), press **`-`** to reduce the effect.

4. Once the line is straight, **write down the k value** printed in the terminal. Hardcode it (e.g. in config or `DISTORTION_K`) for the rest of the project.

5. Optional: adjust **net mask** so the bottom of the frame (net) is excluded:
   ```bash
   python scripts/preprocess.py --net-mask 0.10
   ```
   (0.10 = bottom 10%; default 0.15 = bottom 15%.)
