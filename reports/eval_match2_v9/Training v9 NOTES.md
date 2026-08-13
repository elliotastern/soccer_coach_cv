# Training v9 NOTES

Created 2026-08-12 after v9 train + Match 2 gold P_emit eval finished. This is the writeup for that run.

**Date:** 2026-08-12 / 2026-08-13 UTC  
**Goal:** Phase 1 PoC — **P_emit ≥ 0.80** at conf ≥ **0.80**, IoU 0.5, **n_emitted ≥ 5** (not hollow). Secondary: clear-ball recall ≥ 0.80 (min side ≥ 25 px). Not all-frame recall for a whole match.

**Score vs that goal:** v8 **2/10** → v9 **5/10**. A **10** is P_emit pass **and** clear-ball R ≥ 0.80 @0.8 on held-out clear Match 2 balls (this harvest-50 is optimistic vs a full match).

## What we trained

Resume **v8** (`models/v8_snaps/post_train/checkpoint.pth`, epoch 139) through **epoch 180** on mix pack `ball_finetune_match2_v9`.

Hardware: RunPod A40. SSH: exposed TCP (`root@IP -p PORT` + `~/.ssh/id_ed25519_runpod`). Proxy `ssh.runpod.io` has no scp.

Config: `configs/finetune_match2_v9.yaml`  
Train script: `scripts/train_ball.py` (one-shot RF-DETR resume; `epochs` is absolute `range(start, epochs)`).  
Overnight: `scripts/overnight_v9_match2.sh`  
Sync: `scripts/sync_v9_to_runpod.sh`

Checkpoint (gitignored, saved locally): `models/v9_snaps/post_train/checkpoint.pth` (~475 MB).

## Dataset (never train on Match 2 gold)

Built by `scripts/gold_set/build_ball_finetune_match2_v9.py`. Manifest: `data/processed/gold_sets/ball_finetune_match2_v9_manifest.json`. Pack lives on disk at `/Volumes/LaCie/Projects/Soccer project data/ball_finetune_match2_v9/` (not in git).

| Split | Images | Boxes | Contents |
|---|---:|---:|---|
| train | 996 | 1111 | Match 1 gold + Match 2 train100 (87) ×3 aug, plus light OFFICIAL (~100) and kjoyy (~200) |
| valid | 90 | 92 | 10 Match 2 train100 holdout + stills |
| test (COCO, not PoC) | 85 | 98 | Gold100 strip 0–49 with ball GT (35) + Match 2 gold 50 |

Match 2 train100: 100 labeled frames (Cam5plus / Cam4plus / P10), split 87 train / 10 valid / 3 empty skipped. Gold XML is source of truth, not prelabels.

**Aug:** `expand_match_with_augs` ×3 on **Match 1 and Match 2 together** (145 + 87 = 232 raw → 232+232 augs). Copies are mis-tagged `match1_train_aug*` even when the source is Match 2. Match 2 is ~261/696 match frames after aug, still outnumbered by Match 1.

**Held out (never in train):** `match2_gold_frames` (50) and Gold100 0–49.

## Train issues (and fixes)

1. **Empty train COCO.** `check_coco_dataset_exists` only counted `*.png`, so jpg packs looked empty, YOLO convert overwrote `_annotations.coco.json` (259 bytes, 0 images). RF-DETR loaded 0 train / 90 valid and crashed. Fix: count jpg+png; restore JSON from local pack (`train 996/1111`). Do not re-run YOLO convert.
2. **COCO test AP@0.50** started ~0.83 and ended ~0.83. Pack AP is **not** the PoC metric (v8 already had high pack AP and ~0 emits @0.8).
3. Overnight PoC eval failed: gold pack was not on the pod, and train writes `models/checkpoint.pth` not `models/v9_snaps/post_train/`. Copied gold images+coco, then ran eval by hand.

## How P_emit was measured

```bash
python3 scripts/gold_set/eval_poc_ball_metrics.py \
  --ball-checkpoint models/v9_snaps/post_train/checkpoint.pth \
  --gold-dir data/processed/gold_sets/match2_gold_frames \
  --strip-max 49 --require-ball-gt \
  --out reports/eval_match2_v9/poc_match2_gold_v9.json
```

Same command with v8 ckpt → `poc_match2_gold_v8.json`. SAHI off. Detect floor 0.3, emit bar 0.8, IoU 0.5. Gold coco + `images/` (50 frames, 62 ball boxes; all clear side≥25 on this pack).

**Emit** = publish a ball only if conf ≥ 0.80. **P_emit** = precision of those published boxes. Hollow = P_emit looks high with n_emitted < 5.

## Results — Match 2 gold 50

| ckpt | conf | P_emit | n_emitted | clear-ball R | note |
|---|---|---|---:|---:|---|
| v8 | 0.5 | 1.00 | 43 | 69% | sees the ball |
| v8 | **0.8** | 1.00 | **3** | **5%** | hollow |
| **v9** | 0.5 | 0.97 | 58 | 90% | 2 FP |
| **v9** | **0.8** | **1.00** | **19** | **31%** | real emit pass |

v9 opened the emit gate (3 → 19) with zero FP @0.8. Clear-ball R @0.8 is 31%, not 80%.

## Next (not this commit)

Match-2-heavier mix (×5–6 on Match 2, light Match 1, cut stills) and/or more Match 2 labels. Do not more-epochs this mix. Rank on P_emit + clear-ball R, not COCO AP.
