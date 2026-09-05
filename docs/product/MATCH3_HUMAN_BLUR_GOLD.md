# Match 3 human blur / soft gold

**Canonical export (git):** `data/processed/gold_sets/match3_human_blur_gold/gold/human_labels.json`  
**Rule:** only boxes with UI `human_conf` (Confirm box or drag). Seed/prelabel-only frames are **excluded**.  
**Frames:** `match3_human_blur_gold/frames/` are local copies (gitignored). Source JPGs live under each pack’s `review/frames/`.

**Last saved:** 2026-09-04 — **n=339** · streaky=36 · blurry≈119 · clear=160 (see `manifest.json` for exact counts).

## Why this pack exists

Product clear-ball A/B (P10/P8 strips + holdout) is near ceiling on sharp balls. Residual soft/blur fails need **human** soft/blur boxes. This export is the locked human set for that work — do not drop `human_labels.json` from git.

## Source packs (labels JSON also in git)

| Pack | Focus | Clip | Human boxes (at last export) | Notes |
|------|-------|------|------------------------------:|-------|
| `match3_blur_p1_1300` | P1 | soft_t01300 · 21:40–21:45 | 132 | Dense true soft/blur (low Laplacian); primary blur gold |
| `match3_blur_p1_250` | P1 | soft_t00250 · 4:10–4:15 | 74 | Clear + some streak |
| `match3_blur_p1_soft1500` | P1 | soft_t01500 | 101 | Soft-harvest trajectory |
| `match3_quad_p9_655` | P9 | quad_P9_t00655 · 10:55 | 32 | Midfield / other-cam gold_xy fallback |
| `match3_blur_p1_1100` | P1 | soft_t01100 · 18:20–18:25 | 0 | Opened for review; little/no human yet |

Review: http://127.0.0.1:8877/match3-m1-blur (current blur pack) · hub `/match3-m1`  
Viewer: `bash scripts/run_serve_viewer_8877.sh`

## Refresh export after more labeling

```bash
# After Confirm/Save in the M1 UI:
PYTHONPATH=. python3 scripts/gold_set/export_match3_human_blur_gold.py
# then commit JSON under match3_human_blur_gold/ + source pack labels.json (not frames/)
```

## Use

- **Do** score soft/blur residual A/Bs against `human_labels.json` (or filter `blurry` / `streaky`).
- **Do** keep pack `labels.json` + export JSON on GitHub.
- **Don’t** treat seed-only soft dets as gold.
- **Don’t** expect this alone to replace Match1/2 CVAT train gold; it is a Match 3 soft/blur human bank.

Registry: [VALIDATED_GOLD_REGISTRY.md](VALIDATED_GOLD_REGISTRY.md) · M1 UI: [MATCH3_M1_STRIP.md](MATCH3_M1_STRIP.md)
