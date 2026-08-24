# Model weights (not in Git)

RF-DETR checkpoints are too large for GitHub. **Transfer from the developer** (AnyDesk, USB, or private Hugging Face download).

**Client PC (Catch):** Prefer AnyDesk file copy so the machine needs no developer GitHub login (never put developer GitHub auth on Catch). HF is OK for weights-only download with a scoped read token — see [CATCH_MACHINE_CURSOR_CONTEXT.md](../docs/product/CATCH_MACHINE_CURSOR_CONTEXT.md) and `.cursor/rules/catch_client_credentials.mdc`.

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

## Hugging Face (optional)

Private model repo + write token on **developer Mac**; Catch pulls with `HF_TOKEN` then `unset HF_TOKEN`.

```bash
# Mac (upload once)
export HF_TOKEN=hf_...   # write token — do not paste into chat
export HF_REPO=YOUR_HF_USER/soccer-coach-phase1-weights
bash scripts/push_phase1_weights_hf.sh

# Catch (download)
export HF_TOKEN=hf_...
export HF_REPO=YOUR_HF_USER/soccer-coach-phase1-weights
bash scripts/pull_phase1_weights_hf.sh
unset HF_TOKEN
```

Use `hf` (not deprecated `huggingface-cli`). Keep `huggingface_hub>=0.30,<1` so transformers stays happy.
## Verify after copy

```bash
test -f models/people_after_100_epochs.pth && echo "player ok"
test -f models/v12_hard_snaps/post_train/checkpoint.pth && echo "ball ok"
```

## Other files in this folder

Extra snapshots (`v10_snaps`, `v11_snaps`, …) are training artifacts. **Not required** for Phase 1 product demo.

## Commercial use

Use only weights trained on commercial-safe data per project policy ([PHASE1_SCOPE.md](../docs/product/PHASE1_SCOPE.md)).
