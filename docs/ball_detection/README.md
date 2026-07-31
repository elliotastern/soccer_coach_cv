# Ball detection — training and validation notes

Material previously maintained alongside a separate Hugging Face snapshot now lives in this folder. Use it together with the main pipeline [README.md](../../README.md).

## Where things live

| Topic | Location |
|--------|-----------|
| Resume and fast training YAML | [configs/](../../configs/) (`resume_*.yaml`, `training_fast*.yaml`, RT-DETR `rtdetr_*.yml`) |
| Frame validation (100-frame HTML reports) | [scripts/ball_validation/](../../scripts/ball_validation/) |
| First-100-frames analysis variants | [scripts/experiments/](../../scripts/experiments/) |
| Hugging Face model card (YAML frontmatter) | [docs/huggingface_model_card.md](../huggingface_model_card.md) |
| Push repo to Hub | `python scripts/push_to_huggingface.py --repo-id <org>/<name>` |
| COCO archive download | [DOWNLOAD_INSTRUCTIONS.md](../runbooks/DOWNLOAD_INSTRUCTIONS.md) |

## Index of docs in this folder

- [TRAINING_EVALUATION_REPORT.md](TRAINING_EVALUATION_REPORT.md)
- [TRAINING_PLAN.md](TRAINING_PLAN.md)
- [START_TRAINING.md](START_TRAINING.md)
- [RESUME_CONFIGS_README](../../configs/RESUME_CONFIGS_README.md) (in `configs/`)

See the directory listing for the full set of strategy, MLflow, and epoch notes.
