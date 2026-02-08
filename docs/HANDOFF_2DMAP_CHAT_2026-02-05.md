# Handoff: 2D Map Chat Summary (2026-02-05)

**Purpose:** Pick up tomorrow on a new RunPod rental as if you never left. Everything below happened in one chat session.

---

## What We Did (Chronological)

### 1. Plan / push handoff (earlier context)
- You had commit `48890b0` locally; push failed (no GitHub creds in env).
- We added **Push to GitHub** and **Continue tomorrow** sections to `docs/HANDOFF_2DMAP_HALF_PITCH_AND_VERIFY.md`.
- We committed that doc update as `dd1a0b1`.

### 2. Homography fix (4-point, correct goal/halfway mapping)
- **File:** `scripts/test_2dmap_manual_mark.py`
- **Change:** Replaced `homography_from_marks` so that:
  - **TL** and **TR** are on the **goal line** (x=0), not opposite ends of the pitch.
  - Uses **halfway_line_xy** (Halfway Left, Halfway Right) for the halfway line.
  - **Destination mapping:**
    - TL → (0, 0), TR → (0, h_map)
    - Halfway Left → (w_map, 0), Halfway Right → (w_map, h_map)
  - Call updated to: `H = homography_from_marks(src_corners, w_map, h_map, halfway_line_xy=halfway_line_xy)`.

### 3. Requirements install
- Ran `pip install -r requirements.txt`; failed on system `blinker` (distutils).
- Fixed with: `pip install 'blinker>=1.5.0' --ignore-installed` then re-ran `pip install -r requirements.txt`. **rfdetr** and all deps now installed.

### 4. Serve viewer on port / fix serve_viewer.py
- You wanted to “fire up” port 39449; that port was used by the IDE (Code Extension Host).
- We fixed **`serve_viewer.py`**: bytes literals had a non-ASCII character (en-dash `–`); replaced with ASCII `-`.
- Started viewer on **port 8080** instead. For remote access, add port 8080 in the Ports panel and use the forwarded URL.
- Docstring in `serve_viewer.py` updated: default port 6851 → 8080.

### 5. Remove red center circle on 2D map
- **File:** `scripts/test_2dmap_manual_mark.py`
- Center dot + “Center” label drawing made conditional on `draw_center`.
- Report now calls `_draw_boxes_and_landmarks_on_map(..., draw_center=False)` so the red center is not drawn on the map.

### 6. Y-axis orientation fix (camera near/far = map top/bottom)
- **File:** `scripts/test_2dmap_manual_mark.py` — `homography_from_marks` destination.
- **Bug:** Players far from camera were at top of map; should be at bottom.
- **Fix:** Swapped y so that:
  - TL (near) → (0, 0) = **top**
  - TR (far) → (0, h_map) = **bottom**
  - Halfway Left → (half_w, 0), Halfway Right → (half_w, h_map).

### 7. Halfway line at right edge of half-pitch (spacing fix)
- **Bug:** `half_w = w_map // 2` (262) compressed players into a small area.
- **Fix:** Halfway line at **x = w_map** (525), not `w_map // 2`. So:
  - Halfway Left → (w_map, 0), Halfway Right → (w_map, h_map).
- Players now spread across the full half-pitch width (goal line to halfway).

### 8. Halfway line label positions
- **Bug:** “Halfway Left” was drawn at the **left** edge of the pitch (goal line) instead of on the halfway line.
- **Fix:** When `use_fixed_halfway_positions=True`, both halfway points are on the **same vertical line** (halfway):
  - `halfway_x = margin + w_map - 1`
  - Halfway Left → (halfway_x, margin) — top of pitch
  - Halfway Right → (halfway_x, margin + h_map - 1) — bottom of pitch.

### 9. Report “Weights” label
- **File:** `scripts/test_2dmap_manual_mark.py`
- Report HTML now shows the **actual** weights used (e.g. `checkpoint_best_total_after_100_epochs.pth` or `COCO (no checkpoint)`).
- Added `used_weights_name` and pass it into the HTML.

### 10. GitHub pushes (with your token)
- **Commit 1:** `596ef56` — “working for 2d map for the first time!!!!!!!” (homography, spacing, red center removed, weights label).
- **Commit 2:** `db4d981` — “Breakthrough! 2d map now has amazong spacing for the first time” (halfway at w_map).
- **Commit 3:** `c8b6294` — “breakthrough! 2d mapping now working and diagram looks correct. claude said correct too!!!” (halfway labels on correct line).
- Token was used only for push; remote URL was reset to `https://github.com/elliotastern/soccer_coach_cv.git` after each push.
- **Security:** You were reminded to revoke the token (it was shared in chat) and create a new one.

---

## Key Files

| File | Role |
|------|------|
| **scripts/test_2dmap_manual_mark.py** | Main 2D map report: homography (4-point with halfway_line_xy), y-orientation, halfway at w_map, no red center, halfway labels on correct line, weights label in HTML. Generates the report. |
| **scripts/verify_2dmap_report.py** | Verifies report output (marks, size, right goal, pitch green, spread, players vs picture). |
| **serve_viewer.py** | HTTP server for viewing reports (default port 8080). Fixed ASCII in bytes; docstring says 8080. |
| **docs/HANDOFF_2DMAP_HALF_PITCH_AND_VERIFY.md** | Handoff doc: Option A/B, verification, push instructions, commands. |
| **data/output/2dmap_manual_mark/manual_marks.json** | Saved marks (TL, TR, halfway line, etc.). |
| **data/output/2dmap_manual_mark/test_2dmap_manual_mark.html** | The generated report (first 10 frames, picture left, 2D map right). |

---

## Commands You’ll Use

**Regenerate the 2D map report (no marking UI):**
```bash
cd /path/to/soccer_coach_cv
python scripts/test_2dmap_manual_mark.py --use-saved data/raw/E806151B-8C90-41E3-AFD1-1F171968A0D9.mp4 --half-pitch-style left_half
```

**Report with only the visible half (no right goal):**
```bash
python scripts/test_2dmap_manual_mark.py --use-saved data/raw/E806151B-8C90-41E3-AFD1-1F171968A0D9.mp4 --half-pitch-style half_only
```

**Verify report only:**
```bash
python scripts/verify_2dmap_report.py --half-pitch-style left_half
```

**Serve the report (e.g. on RunPod, then forward port 8080):**
```bash
python serve_viewer.py --port 8080
```
- Open: http://localhost:8080/2dmap (or your forwarded URL)

**Install dependencies (if fresh env):**
```bash
pip install 'blinker>=1.5.0' --ignore-installed
pip install -r requirements.txt
```

---

## Current State

- **Branch:** `main`, in sync with `origin/main` after last push.
- **Latest commit:** `c8b6294` — “breakthrough! 2d mapping now working and diagram looks correct. claude said correct too!!!”
- **Video used:** `data/raw/E806151B-8C90-41E3-AFD1-1F171968A0D9.mp4`
- **Trained model:** `models/checkpoint_best_total_after_100_epochs.pth` (used by default for player bboxes).
- **Uncommitted (optional):** e.g. `docs/VIEWING_2DMAP_REPORT.md`, `tune_distortion_web.py`, various untracked docs/scripts — push was only the 2D map script.

---

## Homography Summary (for reference)

- **Source (image):** TL, TR, Halfway Left, Halfway Right (from marks).
- **Destination (map):**
  - Goal line: x = 0 (TL and TR).
  - Halfway line: x = w_map (525) — right edge of half-pitch.
  - Near touchline: y = 0 (top). Far touchline: y = h_map (bottom).
- So: top = near camera, bottom = far; left = goal, right = halfway.

---

## If You Clone on a New RunPod

1. `git clone https://github.com/elliotastern/soccer_coach_cv.git && cd soccer_coach_cv`
2. Install deps (see above; blinker fix if needed).
3. Ensure video exists: `data/raw/E806151B-8C90-41E3-AFD1-1F171968A0D9.mp4` (or update path in commands).
4. Marks are in `data/output/2dmap_manual_mark/manual_marks.json` — if you don’t copy output dir, run once with `--mark --web` to recreate marks, or copy that file from backup.
5. Run report command above; run `serve_viewer.py`; forward port 8080 and open `/2dmap`.

---

## Possible Next Steps (from handoff / chat)

- Run `--half-pitch-style half_only` if you prefer diagram with only the visible half.
- Tune verification (e.g. relax “players match picture” or spread) in `scripts/verify_2dmap_report.py`.
- Add `--verify` to CI.
- Revoke the old GitHub token and create a new one if you haven’t already.

You’re in a good state to continue 2D map work or move on to the next feature.
