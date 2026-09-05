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

## Ckpt A/B on this bank (report-only, 2026-09-04)

Script: `scripts/gold_set/ab_match3_human_blur_gold_ckpts.py`  
Report: `reports/eval_match3/improve_eng_loop/ab_match3_human_blur_gold_ckpts.json`

Scored v12 / v14–v22 (v13 weights missing on Catch) at thr **0.10**, recall @ IoU≥0.30 / center≤20px on **339** human boxes.

| ckpt | all R@0.30 | blurry R@0.30 (n=119) | streaky R@0.30 (n=36) | clear R@0.30 |
|------|----------:|----------------------:|----------------------:|-------------:|
| v12 | 0.826 | **1.000** | 0.722 | 0.994 |
| v14 | 0.799 | 0.983 | 0.722 | 0.975 |
| v15 | 0.829 | 1.000 | 0.722 | 0.994 |
| **v16** | 0.823 | **1.000** | 0.722 | 0.988 |
| v17–v19 | ~0.82–0.83 | 1.000 | 0.722 | ~0.988 |
| v20 | **0.835** | 1.000 | 0.722 | 0.994 |
| v21 | 0.829 | 1.000 | 0.722 | 0.988 |
| v22 | 0.814 | 1.000 | 0.722 | 0.988 |

**Read:** low-Laplacian “blurry” boxes are already **solved** by product v16 (and peers). The shared miss pool is **streaky** (~28% miss, same across versions). Residuals v17–v22 do **not** lift streak recall here; v20 is a tiny overall bump only. Not a promote signal — keep v16 until streak-specific gold + train changes.

Registry: [VALIDATED_GOLD_REGISTRY.md](VALIDATED_GOLD_REGISTRY.md) · M1 UI: [MATCH3_M1_STRIP.md](MATCH3_M1_STRIP.md)
