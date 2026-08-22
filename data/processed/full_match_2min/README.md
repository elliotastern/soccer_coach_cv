# Phase 1 smoke batch output (bundled for client demo)

~2 minutes of Match 3 processing so **Match Review** works immediately after clone + model weights — no full batch required for first walkthrough.

| Folder | Contents |
|--------|----------|
| `P10-002/` | `frame_data.csv`, `events.json`, `checkpoints/` |
| `P1-006/` | Same (second camera smoke) |

**Dashboard default output root:** `data/output/full_match_2min` (see `src/review/app.py`).

**Mosaic video tiles** still need raw MP4s in `data/raw/Match 3/` (P10, P7, P8, P9). Events and pitch panel work from this CSV/JSON alone.

Regenerate locally:

```bash
python3 apps/batch_pipeline.py --video "data/raw/Match 3/P10-002.mp4" \
  --output data/output/full_match_2min --max-frames 3600
```
