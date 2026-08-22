# Model weights (not in Git)

RF-DETR checkpoints are too large for GitHub. **Transfer from the developer** (AnyDesk, USB, or cloud link).

## Required for Phase 1 (Match Review + batch)

| File | Approx size | Role |
|------|------------:|------|
| `people_after_100_epochs.pth` | ~128 MB | Player detection |
| `v12_hard_snaps/post_train/checkpoint.pth` | ~350 MB | Ball detection (v12 hard) |

Configured in `configs/default.yaml`:

```yaml
player_checkpoint: models/people_after_100_epochs.pth
ball_checkpoint: models/v12_hard_snaps/post_train/checkpoint.pth
```

## Install layout

```text
models/
  people_after_100_epochs.pth
  v12_hard_snaps/
    post_train/
      checkpoint.pth
```

## Verify after copy

```bash
test -f models/people_after_100_epochs.pth && echo "player ok"
test -f models/v12_hard_snaps/post_train/checkpoint.pth && echo "ball ok"
```

## Other files in this folder

Extra snapshots (`v10_snaps`, `v11_snaps`, …) are training artifacts. **Not required** for Phase 1 product demo.

## Commercial use

Use only weights trained on commercial-safe data per project policy ([PHASE1_SCOPE.md](../docs/product/PHASE1_SCOPE.md)).
