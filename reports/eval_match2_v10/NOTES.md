# Training v10 NOTES

Created 2026-08-13 after v10 train, Match 2 gold still PoC, video system eval, and a no-emit-gate recall pass.

**Date:** 2026-08-13  
**Checkpoint (gitignored, local):** `models/v10_snaps/post_train/checkpoint.pth` (~475 MB). Resume from v9 through epoch **205**.

## Two different questions

| Question | Metric | Stack | Result on held-out Match 2 gold 50 |
|---|---|---|---|
| Client PoC “~80% accuracy” | **P_emit** of boxes published at conf ≥ 0.80 | detector, or detector+ByteTrack+Kalman+emit gate | **1.00**, 0 FP, 27–28 emits (not hollow) |
| “Are we getting 80%+ of the balls?” | **Recall** of labeled balls (IoU 0.5), ignore 0.80 gate | **detector only** | **92% @0.3**, **89% @0.5**; precision **97%** |

Those recall numbers are **not** Kalman/ByteTrack. Enhancements **lower** recall on this set (see below).

Gold 50 is harvested large/clear balls (mostly P10, some Cam4plus, one P8). **Cam5plus has no gold labels.** This is not every ball in the match. Train100 was used in training — do not eval on it.

## What we trained

Resume **v9** → epoch **205** (~25 epochs), lower LR (`3e-5` / encoder `2e-5`). Match2-heavy mix so Match 1 / other fields are not wiped.

Config: `configs/finetune_match2_v10.yaml`  
Pack builder: `scripts/gold_set/build_ball_finetune_match2_v10.py`  
Manifest: `data/processed/gold_sets/ball_finetune_match2_v10_manifest.json`  
Pack on disk (not in git): `/Volumes/LaCie/Projects/Soccer project data/ball_finetune_match2_v10/`  
Overnight: `scripts/overnight_v10_match2.sh`  
Sync: `scripts/sync_v10_to_runpod.sh`  
Hardware: RunPod A40.

| Split | Images | Notes |
|---|---:|---|
| train | 825 | Match2 ×5 (435), Match1 ×2 (290), light stills (100) |
| valid | 45 | 10 Match2 train100 holdout + stills |
| test (COCO, not PoC) | 85 | Gold100 0–49 (35) + Match 2 gold 50 |

**Held out (never in train):** `match2_gold_frames` (50) and Gold100 strip 0–49.

`scripts/train_ball.py` now **refuses YOLO convert** if COCO JSON exists with 0 images but jpg/png are present (v9 wipe bug).

## Detector-only recall (no 0.80 gate)

```bash
python3 scripts/gold_set/eval_match2_v10_recall.py
```

Still gold JPEGs. Size filter + top-2. **Kalman=off, ByteTrack=off, SAHI=off.** JSON: `recall_no_emit_gate.json`.

| conf | recall (balls found) | precision | frame hit | tp/fp/fn |
|---|---:|---:|---:|---|
| ≥ 0.3 | **0.919** (57/62) | **0.966** | **50/50** | 57/2/5 |
| ≥ 0.5 | **0.887** (55/62) | **0.965** | **50/50** | 55/2/7 |

By camera @0.3:

| cam | frames | recall | precision |
|---|---:|---:|---:|
| P10 | 36 | 1.00 | 1.00 |
| Cam4plus | 13 | 0.80 | 0.91 |
| P8 | 1 | 1.00 | 1.00 |

Every labeled frame gets at least one correct ball. Misses are mostly extra balls on the 12 two-ball frames. Cam4plus @0.5 drops to **0.72** recall.

**Other field check (Gold100 0–49, 35 ball-present frames):** @0.5 P_emit=0.40 (3 FP); @0.8 **0 emits**. Match2-heavy v10 hurt Match 1 stills.

## Kalman + ByteTrack + emit 0.80

```bash
python3 scripts/gold_set/eval_match2_v10_video_system.py
```

Detect 0.3 → Kalman on → ByteTrack → emit if EMA or raw ≥ 0.80 → Cam5plus/Cam4plus max-conf. JSON: `video_system.json`.

Same gold 50, 10 video warmup frames before each gold JPEG:

| stack | recall | precision | n_emitted |
|---|---:|---:|---:|
| Detector only @0.3 | 0.92 | 0.97 | 59 boxes |
| + Kalman + ByteTrack (raw, no 0.80) | **0.71** | 0.88 | 50 |
| + emit ≥ 0.80 | **0.44** | **1.00** | 27 (0 FP) |

ByteTrack dropped some raw≥0.80 boxes. EMA pulled a few 0.78s over the gate. Net: **enhancements did not find more balls.** Emit gate is working as designed (0 FP, lower recall).

## Product video strip (t=33s, 100 consecutive frames)

Cam5plus + Cam4plus, same window as the v8 100-in-a-row. **No full GT** on this strip (one overlapping gold frame, Cam4plus f1980).

| | v8 @0.80 | v10 emit @0.80 | v10 raw detect 0.3 (best cam) |
|---|---|---|---|
| publishes | 0% | **0 / 100** | 100 / 100, mean conf **0.72**, all Cam4plus |

Detect is alive every frame. The 0.80 gate publishes nothing on this play. Contact: `bestcam_contact_10x10.jpg`.

## How to reproduce

```bash
# detector-only recall (the 92%/89% numbers)
python3 scripts/gold_set/eval_match2_v10_recall.py

# still PoC including 0.80 emit bar
python3 scripts/gold_set/eval_poc_ball_metrics.py \
  --ball-checkpoint models/v10_snaps/post_train/checkpoint.pth \
  --gold-dir data/processed/gold_sets/match2_gold_frames \
  --strip-max 49 --require-ball-gt \
  --out reports/eval_match2_v10/poc_match2_gold.json

# video stack
python3 scripts/gold_set/eval_match2_v10_video_system.py
```

Unit tests (no model): `test_eval_match2_v10_recall.py`, `test_eval_match2_v10_video_system.py`, `test_ball_finetune_match2_v10.py`.

## Read

- **Detector** on this Match 2 gold harvest: yes, **80%+ of labeled balls**, boxes almost all correct.
- **That is not the whole match** (no Cam5plus gold; ordinary t=33s clip unlabeled).
- **Kalman/ByteTrack/0.80 emit** = high precision, low recall. They are not how we got 92%.
- Do not remap 0.70 → 0.80. Do not train on Match 2 gold eval frames or on prelabels.

## 4 quad test (named Match 2 windows)

Replaces the random 5×5 strip for review. Four synced windows, all 8 Match 2 cameras, best-cam `max_conf`, detect 0.3 → ByteTrack → emit 0.80. Dashboard: `http://127.0.0.1:8080/4quad` (`serve_viewer.py`). Script: `scripts/gold_set/run_5x5_ball_clips.py --quad-test`. Pack (videos not in git, ~2.4 GB): `reports/eval_match2_v10/4quad_test/`.

| Slot | Clock | Start | Frames | Selected raw | Emit @0.80 | Mean conf | Top cam |
|---|---|---:|---:|---|---:|---:|---|
| Center Start | 0:08–0:13 | 8s | 300 | 300/300 | 0.7% | 0.775 | Cam4plus |
| Bottom Right | 6:52–6:58 | 412s | 359 | 359/359 | 11.1% | 0.756 | Cam4plus |
| Top Left | 0:26–0:31 | 26s | 299 | 299/299 | 15.7% | 0.790 | Cam4plus |
| Top Right | 2:05–2:10 | 125s | 299 | 299/299 | 22.1% | 0.813 | Cam4plus |

Boxes are drawn on **Mosaic** and **Selected** overlays only. Per-camera filter clips (`source/quad_*_P1.mp4` etc.) are raw extracts with **no** boxes. Overlay mp4s are H.264 so the browser can play them. Layout is full-width stacked clips, not 2×2.

```bash
python3 scripts/gold_set/run_5x5_ball_clips.py --quad-test
python3 scripts/gold_set/test_run_5x5_ball_clips.py
```

## Ball postprocessing test (P10 Top Left) — visual pick

Dashboard: `http://127.0.0.1:8080/ball_postprocessing_test`. Script: `scripts/gold_set/run_ball_postprocessing_test.py`. Pack metadata in git; overlay mp4s stay local (~100 MB).

Ten curated stacks on Match 2 **Top Left 0:26–0:31, P10 only**, ordered from prior train/gold studies. After visual review of that gallery:

**Winner: SAHI fallback-only** (`sahi_fallback`) — full-frame detect first; tiles only when full-frame is empty after the size filter. On this clip: raw hold **78.0%**, emit hold **26.7%** (baseline topk=2 was 50.7% / 15.3%).

Still not a gold rescore. Earlier still ablation treated fallback as identical to baseline because labeled frames already had a det. On this video window tiles actually fire. Next candidates to try as combos: TTA or multiscale on full-frame **plus** SAHI fallback for remaining empties.

## Match2 Top Left 300 gold — 30-stack postproc rank

After human labels on the same P10 window (`match2_4quad_top_left`, 265 GT balls / 300 frames), all 30 gallery stacks were rescored vs gold XML:

`reports/eval_match2_v10/top_left_300_postproc_rank/` · script `scripts/gold_set/eval_top_left_300_postproc_rank.py`

| Rank | Stack | F1@0.3 | R@0.3 | P@0.3 |
|---:|---|---:|---:|---:|
| 1 | `sahi_dense_tiles` | 0.847 | 0.796 | 0.906 |
| 2 | `D7_adaptive_asahi` | 0.821 | 0.789 | 0.857 |
| 3–6 | SAHI-always family (multiscale / topk3 / topk5 / recover-always) | 0.805 | 0.702 | 0.944 |
| 11 | `sahi_fallback` | 0.795 | 0.687 | 0.943 |
| 22 | `baseline_topk2` | 0.681 | 0.536 | 0.934 |
| 30 | `sahi_always_kalman` | 0.267 | 0.253 | 0.283 |

Dense tiles win recall here; product emit @0.8 stays high-P / low-R (~0.20–0.25 R) across the leaders. Spec: [MATCH2_4QUAD_TOP_LEFT_300.md](../../docs/product/MATCH2_4QUAD_TOP_LEFT_300.md).

Mac latency bench (`latency_bench.json`, not 5090): dense median ~**1.55 s/frame** → offline-only; live path stays no-SAHI / fallback.

## 6 P-cam multicam baseline + soft consensus (Top Left window)

Script: `scripts/gold_set/eval_match2_top_left_multicam_baseline.py`  
Reports: `top_left_multicam_baseline/`, `top_left_multicam_consensus/` · proxy gallery `/multicam-proxy`

| Stack | Proxy P (P10-selected) | Proxy R | Who wins max_conf |
|---|---:|---:|---|
| Baseline A max_conf @0.30 | 0.948 | 0.948 | **P7 51%**, P10 38% |
| Soft consensus thr0.15 ≥2 cams | 0.948 | 0.948 | same proxy set; no lift |
| P10 single-cam @0.30 | 0.933 | 0.528 | — |

Gate: proxy OK on P10-win frames → next **5090 latency**; need **P7 gold** (pack `match2_4quad_top_left_p7`, route `/4quad-cvat/top_left_p7`) before claiming full-system R/P. Epipolar blocked (no Match 2 extrinsics). Dense SAHI stays out of live path.

## Dual-gold system score (P7 + P10 labeled)

After human P7 gold: score when selected cam is P7 or P10 vs that cam’s XML.

| Slice (max_conf @0.30) | P | R | vs goal R≥0.8 P≥0.9 |
|---|---:|---:|---|
| **P7∪P10 system** (covered ~89%) | **0.750** | **0.785** | **MISS** (both short) |
| P10-selected only | 0.948 | 0.948 | HIT |
| P7-selected only | 0.601 | 0.652 | MISS — bottleneck |
| Soft consensus (same dual gold) | 0.744 | 0.776 | MISS; no lift |

**Read:** old P10-only proxy overstated the system. P7-selected frames drag P and R below goal. Next: diagnose P7 max_conf FPs/misses (selection vs detect), not latency yet and not dense SAHI for live.

## Selection fix loop (cache only)

Script: `scripts/gold_set/eval_top_left_multicam_selection_loop.py`  
Report: `reports/eval_match2_v10/top_left_multicam_selection_loop/ranking.md`

Raising conf clears the dual-gold proxy goal. Practical pick: **`p7_thr060_others030`** (P7 must be ≥0.60 to compete; others stay @0.30) → system **P=R=0.915**, HIT, 188 covered frames. Plain `max_conf_070` is stronger P/R but only 120 covered. Prefer-P10 knobs did not help at 0.30. Locked in `multicam_select_policy.py` (**Top Left / P-cam study only**).

## 4quad region survey (8 cams, no gold R/P)

Script: `scripts/gold_set/eval_match2_4quad_multicam_survey.py`  
Report: `reports/eval_match2_v10/4quad_multicam_survey/survey.md`

When **Cam4plus/Cam5plus** are in the pool, Top Left’s winner flips to **Cam4plus ~79%** (P10 ~21%) — the earlier P7/P10 dual-gold study was **P-cam-only**. Other regions:

| Slot | #1 @0.30 | Share |
|---|---|---:|
| Top Left | Cam4plus | ~79% |
| Top Right | Cam4plus | ~47% (P7 ~29%, P1 ~23%) |
| Center Start | Cam4plus | ~82% |
| Bottom Right | (see survey.md) | |

Top Left’s `P7≥0.60` floor does **not** change these shares (winners already high-conf). **Label pack ready:** `match2_4quad_center_start_cam4plus` · http://127.0.0.1:8080/4quad-cvat/center_start_cam4plus (dense prelabels; human-correct before GT). Alt later: Top Right (more mixed).

