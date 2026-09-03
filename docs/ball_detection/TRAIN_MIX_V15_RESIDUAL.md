# Ball finetune mix v15 — residual FN after v14 (resume-from-v14)

**Builder:** `scripts/gold_set/build_ball_finetune_v15_residual.py`  
**Pack:** `/Volumes/LaCie/Projects/Soccer project data/ball_finetune_v15_residual/`  
**Manifest:** `data/processed/gold_sets/ball_finetune_v15_residual_manifest.json`  
**Config (Catch):** `configs/finetune_v15_residual_catch.yaml`  
**Resume:** `models/v14_residual_snaps/post_train/checkpoint.pth` @ epoch **275** → absolute end **285** (~10 specialty epochs @560)

Leftover soft-conf / no-det on Match 3 strips scored with **v14** det caches. Hard blur ×3 + tiny paste ×3 on residual only. No holdout-gallery frames.

## Never train

Match 3 frames **120–194**; holdout gallery clips.

## Gates

Strip `P_emit` ≥ 0.80; holdout clear-ball proxy R ≥ gate (~0.884); do not lower emit / widen agree.

## Catch train

```bash
# Mac: build pack, push weights+pack, git push
PYTHONPATH=. python3 scripts/gold_set/build_ball_finetune_v15_residual.py

# Catch (after rsync):
cd ~/soccer_coach_cv && git pull
source ~/.venvs/soccer-rfdetr312/bin/activate
tmux new -s v15ball
PYTHONPATH=. python3 -u scripts/train_ball.py --config configs/finetune_v15_residual_catch.yaml
```
