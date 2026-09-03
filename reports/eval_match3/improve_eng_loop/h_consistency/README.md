# H consistency (Phase 0–1)

## Phase 0 baseline

```bash
PYTHONPATH=. python3 scripts/gold_set/report_match3_h_consistency.py
```

Writes `h_consistency_baseline.json`: landmark round-trip per cam + holdout pairwise mapped-ball span.

- If `span_median_m ≤ 2.5` → H not bottleneck.
- Current baseline: span median ≫ 2.5 m → improve calib before shrinking `AGREE_M`.

## Phase 1 H inputs

```bash
# Dry-run: re-DLT from existing clicks (no write)
PYTHONPATH=. python3 scripts/gold_set/refit_match3_h_from_landmarks.py

# Pitch-1 auto-H spike (OpenCV white-line intersections + seed H snap)
PYTHONPATH=. python3 scripts/gold_set/auto_match3_pitch1_h.py --mode seed
# → pitch1_auto_h.json  (writes calibs only with --write and RT improvement)

# Learned detector contract later:
#   image_points + pitch_points (Pitch 1 meters)
#   → fit_h_from_paired_points → write_calib_h(..., source="auto_keypoints")
```

Match 4 uses the same P-code `*_manual.json` files. Do not import Match 2 FIFA auto-H.

### Classical spike result (seed mode)

OpenCV line crossings under-match fisheye quads (often fewer than 4 snaps). Goal cams can refresh; product H stays manual until a Pitch-1 keypoint model (HRNet-class, commercial-use-safe) raises match count and RT.
