# Path to PoC 80% — from Match 2 100-in-a-row

**PoC bar** ([PHASE1_SCOPE.md](../../docs/product/PHASE1_SCOPE.md)): precision of **emitted** ball preds **P_emit ≥ 0.80** at conf ≥ **0.80**, IoU 0.5. Secondary: clear-ball recall ≥ 0.80 (min side ≥ 25 px). Not all-frame R@0.8.

**Dashboard:** `python3 serve_viewer.py` → [http://127.0.0.1:8080/match2-100row](http://127.0.0.1:8080/match2-100row)

## What this strip showed (t=33s, 100 frames, v8)

| Rank | Cam | @0.5 | @0.8 | median side | max conf |
|------|-----|------|------|-------------|----------|
| 1 | **Cam5plus** | 44% | **0%** | **72 px** | 0.66 |
| 2 | **Cam4plus** | 25% | **0%** | 32 px | **0.70** |
| 3 | P10 | 10% | 0% | 33 px | 0.64 |
| 4 | P8 | 2% | 0% | 30 px | 0.55 |
| — | P1/P6/P7 | 0% @0.5 | 0% | ~27–37 when low-conf hits | ≤0.48 |
| — | P12 | no dets | 0% | — | — |

Match 1 baseline on the same scorecard: also **0% @0.8**; at best 2% @0.5 (cam9).

### Read

1. **Size is largely solved** on the better Match 2 angles (clear-ball proxy ≥25 px; Cam5plus often ~70 px). New settings + closer zoom did their job.
2. **Emit gate is not** — zero frames crossed conf ≥ 0.8 on any cam. Same bottleneck as Gold PoC v8 (hollow / no emit @0.8).
3. **Master-cam candidates for Phase 1:** Cam5plus and Cam4plus (then P10). Prefer these for labeling and product emit, not weak end-line views on this play.

## Concrete sequence to hit 80%

1. **Pick winners in the dashboard** — scrub Cam5plus / Cam4plus / P10; confirm boxes are real balls (presence ≠ precision).
2. **Label Match 2 clear-ball GT** on this 100-row strip (and expand to a stratified Match2 Gold). Needed for real **P_emit**, not presence proxies. Viewer stays read-only for now; reuse Gold100 editor when labeling.
3. **Domain finetune (v9)** — mix Match 2 clear balls (1/1000 + closer zoom domain) into the ball train pack. Rank with:
   ```bash
   python3 scripts/gold_set/eval_poc_ball_metrics.py \
     --ball-checkpoint <ckpt> --strip-max 49 --require-ball-gt \
     --out reports/poc_v9_ball.json
   ```
   Re-run this 100-row test after each candidate. **Success:** **P_emit ≥ 0.80** @0.8 **and** **n_emitted ≥ 5** (non-hollow) on labeled clear frames; clear-ball R trending toward ≥0.80.
4. **Product stack** — detect floor ~0.3 → ByteTrack → **emit_thresh 0.80**. Turn SAHI on only if it raises clear-ball recall without flooding FPs ([TRAIN_CHECKPOINT_SELECTION.md](../../docs/ball_detection/TRAIN_CHECKPOINT_SELECTION.md)).
5. **Do not chase** all-frame R@0.8 on far/tiny/occluded balls for Phase 1 — drop below ~0.8 rather than guess.

## Honest status

Hardware/settings cleared the **size** bar on good angles. Hitting client **~80%** now means **raising calibrated confidence on clear balls** via Match-2-domain labels + finetune, then measuring true **P_emit** — not waiting for more shutter tweaks alone.
