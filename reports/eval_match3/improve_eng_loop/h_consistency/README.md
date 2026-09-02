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

# Auto-keypoint contract (commercial-use-safe detector):
#   image_points + pitch_points (Pitch 1 meters from pitch1_landmarks)
#   → fit_h_from_paired_points(...)
#   → write_calib_h(path, H, source="auto_keypoints")
```

Match 4 uses the same P-code `*_manual.json` files. Do not import Match 2 FIFA auto-H.

Plain re-DLT / RANSAC on current clicks does **not** reduce RT (clicks themselves inconsistent with a single plane H on wide FOV). Next H lift needs better landmarks or a Pitch-1 keypoint model — not fuse rewrites.
