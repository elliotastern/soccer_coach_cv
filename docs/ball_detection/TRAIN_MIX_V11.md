# Ball finetune mix v11

**Builder:** `scripts/gold_set/build_ball_finetune_v11.py`  
**Pack:** `/Volumes/LaCie/Projects/Soccer project data/ball_finetune_v11/`  
**Config:** `configs/finetune_v11.yaml`  
**Resume:** v10 snap (single-pass RF-DETR, live-budget path)

Uses **every validated human ball pack**, minus official eval strips. Dense 60 fps clips are **strided** so duplicates don’t swamp Match1/Match2 diversity.

## Train

| Source | What goes in | Why |
|--------|----------------|-----|
| `math_1_training` + `math_1_training_batch3` | All ball frames | Validated Match 1 train |
| `match2_train_label100` train split | 87 frames (existing split) | Validated Match 2 train |
| Top Left 300: P10, P7, Cam4plus | Frames **0–239**, stride **5** | Largest validated set; last 1 s held out |
| Match 3 `match3_quad_p10_31` | Boxes on frames **≤119**, stride **2** | New Pitch 1 / P10 domain |

Light OFFICIAL + kjoyy stills (same counts as v10) so other fields don’t vanish. Match1 ×2 aug, Match2 ×3, dense/Match3 ×3.

## Never train

| Pack | Role |
|------|------|
| `match1_1_100` strip 0–49 | Gold100 PoC eval |
| `match2_gold_frames` / `accepted50` | Match 2 large-ball eval |
| Match 3 frames **120–194** | Honest M1 `P_emit` / clear-ball R |
| 4quad frames **240–299** | Temporal holdout of Top Left gold |
| `math_1_training_batch2` | Not validated |
| `match2_4quad_center_start_cam4plus`, `match2_4quad_label` | Auto / draft |

## Test split in the pack

Gold100 0–49 + `match2_gold_frames` (copied into `test/`). M1 tail is scored in-place via `score_match3_ball_m1.py`, not duplicated into this pack.

## RunPod

```bash
python3 scripts/gold_set/build_ball_finetune_v11.py
# rsync ball_finetune_v11 + configs/finetune_v11.yaml + v10 ckpt
```
