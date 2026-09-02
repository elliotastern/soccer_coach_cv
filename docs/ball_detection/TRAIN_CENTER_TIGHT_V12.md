# RF-DETR center-tight ball boxes (YOLOopt-style)

Apply SoccerNet-v3D lesson without switching off RF-DETR.

## Recipe

1. Build manifest: `python3 scripts/gold_set/build_ball_finetune_center_tight.py`
2. Finetune from `models/v12_hard_snaps/post_train/checkpoint.pth` using `bbox_tight` (24 px square on ball center).
3. A/B on Catch: quad det funnel + `ab_match3_fuse_3d.py` agree rate.

## Gates

- Strip P_emit and clear_R must not drop vs v12_hard baseline.
- 3D fuse agree_among_emit should rise if box feet align with ball center.

See `configs/finetune_v12_hard.yaml` for training hyperparameters; swap label column to `bbox_tight`.
