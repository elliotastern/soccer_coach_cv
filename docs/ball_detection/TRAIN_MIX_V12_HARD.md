# Ball finetune mix v12 — hard small + blur

**Builder:** `scripts/gold_set/build_ball_finetune_v12_hard.py`  
**Pack:** `/Volumes/LaCie/Projects/Soccer project data/ball_finetune_v12_hard/`  
**Config:** `configs/finetune_v12_hard.yaml`  
**Resume:** v11 snap → epochs **230 → 255** (~25 epochs)

Specialty after v11 for the hardest clear-ball misses: **tiny** and **blurred** balls. Same holdouts as v11.

## Strategy

| Layer | What |
|-------|------|
| Anchor | v11 validated Match1 / Match2 / dense / stills (light augs) |
| Hard blur | Frames with ball max-side ≤ **28 px** ×3: Gaussian/Box blur + JPEG crush (+ optional shrink→expand) |
| Tiny paste | Per raw frame ×2: erase real ball, paste **8–18 px** blurred crop (FN-style, like v8 but smaller) |

Train-time RF-DETR augs also keep `motion_blur` + JPEG on.

## Never train

Same as v11: Gold100 0–49, Match2 gold, Match3 **120–194**, 4quad **240–299**, unvalidated packs.

## RunPod

```bash
python3 scripts/gold_set/build_ball_finetune_v12_hard.py
bash scripts/sync_v12_hard_to_runpod.sh
```
