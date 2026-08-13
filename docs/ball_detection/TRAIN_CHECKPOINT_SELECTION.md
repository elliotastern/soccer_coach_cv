# Ball train checkpoint selection (Gold / PoC)

Policy for ranking ball RF-DETR runs. Complements pack COCO val that RF-DETR prints each epoch (diagnostics only).

## Do not interrupt mid-run for this

If a finetune is already running and healthy, **let it finish**. Wire Gold/PoC into the *next* train job, or run offline selection when weights land. Restarting a mid-epoch job only to add eval is not worth it.

## Two bars — do not confuse them

| Bar | What | When | Role |
|---|---|---|---|
| **Pack valid COCO** (RF-DETR built-in) | AP on tiny-FN / train-pack `valid` | After **every** train epoch | Fast train sanity only. Often inflated; **not** the PoC bar. |
| **Gold det PoC** | `P_emit` @ conf ≥ 0.8 (IoU 0.5) + clear-ball recall on **Gold100 strip 0–49**, frames with **≥1 ball** GT | End of run; every **N=5** epochs on future multi-epoch jobs | **Checkpoint selection** among `.pth` files |
| **System product PoC** | Detect (SAHI) → Kalman → ByteTrack emit gate → multi-cam selection | After checkpoint pick, on continuous multi-cam pack | **Product acceptance signal** (n_emitted + P_emit). Entry: `scripts/gold_set/eval_system_ball_poc.py` |

**Hollow pass:** `P_emit ≥ 0.80` with **n_emitted < 5** is not a credible pass — always report n_emitted.

Canonical product metric defs: [PHASE1_SCOPE.md](../product/PHASE1_SCOPE.md), [GOLD100_PLAYER_BALL.md](../product/GOLD100_PLAYER_BALL.md).

## Gold slice for selection

- Pack: `data/processed/gold_sets/match1_1_100/`
- Strip **0–49 only** (held-out for train; never train on this pack)
- Prefer frames with **≥1 ball** annotation for ranking tables (empty-ball frames still matter for FPs if scoring emit precision globally — PoC script uses strip 0–49 as configured)
- Clear-ball: min side ≥ 25 px (full-res)

## Ranking rule (best checkpoint)

When comparing candidates (e.g. `checkpoint_best_regular.pth`, `checkpoint_best_ema.pth`, epoch snaps, final `checkpoint.pth`):

1. Prefer higher **P_emit** at conf ≥ 0.8 (need **≥ 1 emit** or mark hollow / non-comparable).
2. Tie-break: higher **clear-ball recall** @ conf ≥ 0.8.
3. Tie-break: higher pack val AP50 only if still tied (weak signal).
4. Never pick a ckpt solely because pack AP said ~0.9.

Also report SAHI-on PoC as a second table; product default may keep SAHI off unless it improves P_emit without flooding FPs.

## Frequency (next trains)

| Step | Action |
|---|---|
| Every train epoch | Keep RF-DETR pack val (automatic). |
| Every **5** epochs (or on `best_*` file mtime change) | Optional offline Gold PoC on current best + latest (do not block train if GPU is saturated — queue to second GPU/host). |
| **End of run** | Always: snap → `models/v{N}_snaps/`, Gold PoC vs previous baseline (e.g. v6_epoch_110), write `reports/poc_<tag>_ball.json` (+ `_sahi.json`). |
| **After end of run** | System PoC: `eval_system_ball_poc.py` on multicam 20s (SAHI+Kalman+track+selection). Write `reports/system_poc_*.{json,md}`. |

v8 tiny-FN finetune completed; next product read is **system** PoC, not another pack-only train.

## Commands

End-of-run / offline:

```bash
# After weights are local (or scp from RunPod into models/v8_snaps/...)
python scripts/gold_set/select_checkpoint_by_gold_poc.py \
  --gold-dir data/processed/gold_sets/match1_1_100 \
  --strip-max 49 \
  --require-ball-gt \
  --checkpoints models/v8_snaps/post_train/checkpoint.pth \
                models/v6_snaps/epoch_110/checkpoint_best_regular.pth \
  --out reports/poc_v8_vs_v6_select.json

# Wait for remote train then pull + PoC (local machine with gold images)
bash scripts/gold_set/wait_v8_then_poc.sh
```

Single checkpoint Gold tables:

```bash
python scripts/gold_set/eval_poc_ball_metrics.py \
  --ball-checkpoint models/v8_snaps/post_train/checkpoint.pth \
  --strip-max 49 --require-ball-gt \
  --out reports/poc_v8_best_ball.json

python scripts/gold_set/eval_poc_ball_metrics.py \
  --ball-checkpoint models/v8_snaps/post_train/checkpoint.pth \
  --strip-max 49 --require-ball-gt --use-sahi \
  --out reports/poc_v8_best_ball_sahi.json
```

**System product PoC** (track + SAHI + Kalman + multi-cam selection):

```bash
python scripts/gold_set/eval_system_ball_poc.py \
  --pack-dir data/processed/multicam_20s_match1 \
  --ball-checkpoint models/v8_snaps/post_train/checkpoint.pth \
  --out reports/system_poc_v8_multicam20s.json
```

Rebuild multicam detect with SAHI (optional):

```bash
python scripts/gold_set/build_multicam_20s_eval.py \
  --skip-extract --ball-checkpoint models/v8_snaps/post_train/checkpoint.pth \
  --use-sahi --use-kalman --min-thr 0.30
```

## Constraints

- **Do not train** on Gold100 strip 0–49.
- Using Gold only for *selection* can mild-overfit picks to that slice — acceptable for Phase 1 PoC; for longer R&D keep a fixed report slice or ablate N infrequent.
- Prefer commercial-use-safe Match data only.
