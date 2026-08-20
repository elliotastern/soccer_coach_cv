# TrackNet sequence pack v1 (continuous gold)

**Builder:** `scripts/gold_set/build_tracknet_seq_pack.py`  
**Pack:** `/Volumes/LaCie/Projects/Soccer project data/ball_tracknet_seq_v1/`  
**Purpose:** Side A/B for TrackNetV4 / VballNet trained **only** on our continuous, human gold — not RF-DETR COCO mix.

## Sources (dense 60 fps)

| Clip | Camera | Labels | Frames |
|------|--------|--------|--------|
| `match2_4quad_top_left` | P10 | `gold/annotations.xml` | 300 |
| `match2_4quad_top_left_p7` | P7 | `gold/annotations.xml` | 300 |
| `match2_4quad_top_left_cam4plus` | Cam4plus | `gold/annotations.xml` | 299 |
| `match3_quad_p10_31` | P10 | `labels.json` (human-reviewed) | 299 |

Registry: [VALIDATED_GOLD_REGISTRY.md](../product/VALIDATED_GOLD_REGISTRY.md).

**Excluded on purpose:** `match3_quad_p8_87` (not human-reviewed), sparse Match1/Match2 packs, Gold100, `match2_gold_frames`.

## Splits (same temporal holdouts as v11)

| Split | 4quad mid-frame | Match3 P10 mid-frame |
|-------|-----------------|----------------------|
| train | 1–219 | 1–99 |
| valid | 220–239 | 100–119 |
| test | 240+ | 120–193 |

Each row is a **triplet** `(prev, mid, next)`. Visibility can be 0 (empty mid) — useful negatives.

Unlike v11 (stride 5 / 2 for RF-DETR diversity), this pack keeps **every consecutive mid** in-range so TrackNet sees real motion.

## Layout

```text
ball_tracknet_seq_v1/
├── manifest.json
├── clips/<clip_id>/
│   ├── frames/0000.jpg …   # symlinks to gold review JPGs
│   └── labels.json         # per-frame visible / cx,cy / bbox
├── splits/
│   ├── train_triplets.jsonl
│   ├── valid_triplets.jsonl
│   ├── test_triplets.jsonl
│   ├── train_tracknet.csv  # prev,mid,next,visible,x,y
│   ├── valid_tracknet.csv
│   └── test_tracknet.csv
└── heatmaps/{train,valid,test}/<id>.npy   # 512×288 uint8
```

Coords are on the review JPGs (typically 1920×1080). Heatmaps are centered Gaussians (`sigma=2.5`) at that resolution scaled to 512×288.

## Build

```bash
python3 scripts/gold_set/build_tracknet_seq_pack.py --clean
# skip .npy heatmaps if you only want CSV/jsonl:
python3 scripts/gold_set/build_tracknet_seq_pack.py --clean --no-heatmaps
```

## RunPod side train

```bash
# upload pack + start 60-epoch TrackNetV2-style train on 4090
bash scripts/sync_tracknet_seq_to_runpod.sh
# monitor
ssh runpod-tcp 'tail -f /workspace/soccer_cv_ball/reports/tracknet_seq_v1.log'
```

Trainer (PyTorch, from-scratch on this pack only — no tennis weights):

- `scripts/tracknet/model.py` — TrackNetV2-style U-Net (9-ch → heatmap)
- `scripts/tracknet/train.py` — train/valid; peak-within-4px metric
- Checkpoints: `models/tracknet_seq_v1/{best,last}.pth` (pulled from RunPod)

Official TrackNetV4 is TensorFlow; this spike uses the same multi-frame heatmap idea in PyTorch on the existing pod image.

## Results (2026-08-20)

60 epochs on 4090 (~25 min). Weighted MSE required (plain MSE collapsed to empty maps).

| Split | Peak within 4 px on 512×288 |
|-------|----------------------------:|
| best valid | **0.987** (78/79) |
| test holdout | **0.679** (169/249) |

Same test mids vs **v11** (`compare_vs_v11.py`), tol ≈ **15 px** full-res:

| Model | Recall (visible mids) |
|-------|----------------------:|
| TrackNet seq | **0.679** |
| v11 @ conf ≥ 0.80 | 0.293 |
| v11 @ conf ≥ 0.30 | **0.743** |

**Gated** TrackNet (peak heat gate, `gate_sweep_vs_v11.py`) — closest to product “emit or do nothing”:

| Gate | P_emit proxy | clear_R |
|------|-------------:|--------:|
| TN peak ≥ **0.20** (pick for P≥0.80) | **0.810** | **0.651** |
| TN peak ≥ 0.50 | 0.855 | 0.582 |
| v11 emit ≥ 0.80 | **1.000** | 0.293 |

So gated TrackNet ≈ **2.2×** v11 emit recall on this holdout at similar precision floor. Still short of PoC clear_R 0.80; v11@0.30 remains highest raw localization recall.

**Recommendation:** keep **v11 as product ball detector**; treat TrackNet as optional Phase-2/offline recover path on continuous cams — not a replace yet (needs conf calibration, box feet mapping, 8-cam fuse, latency).

Match3 P10 holdout: TN = v11@0.3 = **0.70**. P7 is weakest for TN (0.29). TrackNet always emits a peak (no 0.80 gate).

## Fair A/B vs v11

1. Train TrackNet from scratch on `splits/train_*` (this pack only).
2. Score **test** mid-frames: distance to `(cx,cy)` / heatmap peak.
3. Compare same mids through **v11** RF-DETR — report emit@0.80 and thr@0.30.
4. Match3 test mids 120–193 stay the honest M1 temporal holdout.

```bash
PYTORCH_ENABLE_MPS_FALLBACK=1 python3 scripts/tracknet/compare_vs_v11.py
```

## Notes

- This is a **spike pack**, not a product default detector swap.
- Pack is small (~1k train triplets). Valid ≫ test → some clip overfitting; test is the number that matters.
- CSV paths are relative to the pack root for easy RunPod upload of the whole folder.
