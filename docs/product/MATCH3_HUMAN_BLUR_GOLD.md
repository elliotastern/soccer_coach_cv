# Match 3 human blur / soft gold

**Canonical export (git):** `data/processed/gold_sets/match3_human_blur_gold/gold/human_labels.json`  
**Rule:** only boxes with UI `human_conf` (Confirm box or drag). Seed/prelabel-only frames are **excluded**.  
**Frames:** `match3_human_blur_gold/frames/` are local copies (gitignored). Source JPGs live under each pack’s `review/frames/`.

**Last saved:** 2026-09-05 — **n=404** · streaky=53 · blurry=161 · clear=196 (see `manifest.json` for exact counts).

## Why this pack exists

Product clear-ball A/B (P10/P8 strips + holdout) is near ceiling on sharp balls. Residual soft/blur fails need **human** soft/blur boxes. This export is the locked human set for that work — do not drop `human_labels.json` from git.

## Source packs (labels JSON also in git)

| Pack | Focus | Clip | Human boxes (at last export) | Notes |
|------|-------|------|------------------------------:|-------|
| `match3_blur_p1_1300` | P1 | soft_t01300 · 21:40–21:45 | 132 | Dense true soft/blur (low Laplacian); primary blur gold |
| `match3_blur_p1_250` | P1 | soft_t00250 · 4:10–4:15 | 96 | Clear + streak (session labeled) |
| `match3_blur_p1_soft1500` | P1 | soft_t01500 | 101 | Soft-harvest trajectory |
| `match3_quad_p9_655` | P9 | quad_P9_t00655 · 10:55 | 32 | Midfield / other-cam gold_xy fallback |
| `match3_blur_p1_1100` | P1 | soft_t01100 · 18:20–18:25 | 43 | Soft/blur priority + thin streak |
| `match3_blur_p7_1162` | P7 | pitchmap rand | 0 | Export-listed; label if more streak needed |

Review: http://127.0.0.1:8877/match3-m1-blur → **`match3_blur_p1_1100`** · hub `/match3-m1`  
Viewer: `bash scripts/run_serve_viewer_8877.sh`

### Post-v24 → v25 (real streak paste)

Synth arc (v23 / v23b / v24) did **not** lift streaky R past 0.75. Soft1500’s 9 hard misses are already human-labeled — do not re-synth them. Frozen bank **n=404 / streaky=53**. Specialty train is **v25** (`build_ball_finetune_v25_streak_real_aug.py`): real streak crop paste + elong affine + mild k 5–13 on non-streak soft only.

```bash
PYTHONPATH=. python3 scripts/gold_set/export_match3_human_blur_gold.py
PYTHONPATH=. python3 scripts/gold_set/build_ball_finetune_v25_streak_real_aug.py
bash scripts/push_v25_streak_real_aug_catch.sh
# after ckpt:
PYTHONPATH=. python3 scripts/gold_set/ab_match3_human_blur_gold_ckpts.py --ckpts v16 v25
# if streak gate passes:
PYTHONPATH=. python3 scripts/gold_set/ab_v25_streak_real_aug_vs_v16.py
```

Queue (approx): **1100** leftover streak · **P7_1162** 18 streak — only if v25 fails and autopsy asks for more morphology.

**v25 result (2026-09-05):** Catch train completed; streaky R@0.30 **flat 0.8113** vs v16 (identical 10-miss set). No promote. Product stays **v16**. Autopsy: `streaky_autopsy_v25_flat.json`.

**Next (v26):** Confirm seed boxes on P7_1162 / P7_627 / 1100 leftover (viewer **Confirm box** / Enter), Save, then:

```bash
bash scripts/continue_after_streak_labels.sh
# after ckpt:
PYTHONPATH=. python3 scripts/gold_set/ab_match3_human_blur_gold_ckpts.py --ckpts v16 v26
```

v26 = real-paste with **leave-20% held-out streaky** (no train-on-eval). Gate held-out streaky ≥ max(0.85, baseline+0.05) + product kill.

## Refresh export after more labeling

```bash
# After Confirm/Save in the M1 UI:
PYTHONPATH=. python3 scripts/gold_set/export_match3_human_blur_gold.py
# then commit JSON under match3_human_blur_gold/ + source pack labels.json (not frames/)
```

## Ckpt A/B on this bank (report-only)

Script: `scripts/gold_set/ab_match3_human_blur_gold_ckpts.py`  
Report: `reports/eval_match3/improve_eng_loop/ab_match3_human_blur_gold_ckpts.json`

Historical table (2026-09-04, **n=339 / streaky=36**):

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

**Read:** low-Laplacian “blurry” boxes are already **solved** by product v16 (and peers). The shared miss pool is **streaky**. Re-baseline v16 on the frozen **n=404** bank before judging v25.

## Specialty train gate (v25 real streak paste)

1. Train: Catch `bash scripts/push_v25_streak_real_aug_catch.sh`.
2. Specialty gate: `PYTHONPATH=. python3 scripts/gold_set/ab_match3_human_blur_gold_ckpts.py --ckpts v16 v25` — go if **streaky R@IoU0.30** ≥ **0.80** (or ≥ v16 baseline on same bank + 0.05) and blurry/clear hold.
3. Product kill: `ab_v25_streak_real_aug_vs_v16.py` (strip P_emit / clear_R + holdout). Promote only if both pass.

### Frozen synth results (do not repeat)

| try | streaky R@0.30 | note |
|-----|---------------:|------|
| v16 | 0.722 | product baseline (n=36 era) |
| v23 balanced synth | 0.750 | +0.028, under gate |
| v23b streak-heavy | 0.722 | flat / regress vs v23 |
| v24 miss-core | **0.722** | flat; same 9–10 hard misses |

**Freeze further no-new-label kernel synth.** v25 uses **real streak crops**.

Registry: [VALIDATED_GOLD_REGISTRY.md](VALIDATED_GOLD_REGISTRY.md) · M1 UI: [MATCH3_M1_STRIP.md](MATCH3_M1_STRIP.md)
