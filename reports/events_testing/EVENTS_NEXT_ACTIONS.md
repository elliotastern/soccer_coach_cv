# Events — next actions (live)

Updated as delivery runs. Gates: eng-loops **PASS** · fuse 15s **P_emit 1.0**.

## Catch (running)

| Job | tmux | Monitor |
|-----|------|---------|
| 5-min mosaic (`team_core`) | ~~mosaic5~~ | **DONE** — pulled, 300s, ball_frac 0.95, **54 emits** |
| Full batch P10+P1 | `match4_full` | still running |
| Fuse eval @49s render | `fuse_eval_queue` | timeline **DONE** P_emit 0.67 · MP4 rendering (~30/225) |
| Fuse eval @69s | same queue | after @49s MP4 |

**Scrub now:** `reports/eval_match3/improve_eng_loop/match4_5min/coach_mosaic_pitch_5min.mp4`  
**Rate sheet:** [COACH_RATE_SHEET_mosaic_5min_fuse.md](COACH_RATE_SHEET_mosaic_5min_fuse.md) (54 fuse emits)  
**Holdout clips (stride-15 extract):** `real_fuse_eval_49s/coach_mosaic_stride15_44-64s.mp4` · `real_fuse_eval_69s/coach_mosaic_stride15_69-84s.mp4`

When mosaic completes → auto-staged to `~/soccer_exchange/from_catch/` (`mosaic_watch`).

## Mac commands

```bash
bash scripts/catch_status.sh
bash scripts/pull_match4_full_from_catch.sh
bash scripts/pull_catch_mosaic.sh              # live mosaic (incomplete until RENDER_DONE)
bash scripts/pull_from_catch.sh                # staged MP4s from exchange
bash scripts/push_gold_set_to_catch.sh         # manifest + labels before Catch fuse builds
```

## Labeling / scrub (do now on Mac)

1. Streamlit <code>:8501</code> Expert → `data/output/match_4_5min` → P10
2. Hub: [COACH_SCRUB_HUB.html](COACH_SCRUB_HUB.html)
3. Rate sheets: [P10](COACH_RATE_SHEET_match_4_5min__P10-match4.md) · [P8](COACH_RATE_SHEET_match_4_5min__P8-match4.md)
4. **Read first:** [FUSE_VS_BATCH_FINDING.md](FUSE_VS_BATCH_FINDING.md) — fuse mosaic had 54 typed emits vs batch 16 pass-only

Background: `catch_wait_mosaic_pull.sh` + Catch `mosaic_watch` loop staging on RENDER_DONE.

## Labeling queue (from batch clusters — verify on fuse first)

Match 4 5-min P10 clusters (pass-only, conf≥0.8): ~48–83 s windows — see `suggest_fuse_gold_windows.py` output.

Pick **one** unused window → `build_fuse_eval_window.py` + handover labels (holdout).

## Fuse eval @49s (`real_fuse_eval_49s`) — **coach review**

| Source | Window | Emits |
|--------|--------|-------|
| Batch P10 single-cam | 49.0–51.4 s | **3 pass** (conf 1.0) |
| Product fuse stride 4 | same window | **movement** ~1.8s, **dribble** ~3.3s, movement ~6.8s |

Scored with fuse-truth labels: **P_emit 0.67** (2 tp, 1 fp on late movement). **Not a primary gate** — report-only holdout.

Primary gates still **PASS** (`real_fuse_15s`, holdout, eval@20s).

## Done when

- [x] 5-min mosaic reviewed with fused events bar — **MP4 ready**; fill rate sheet
- [ ] Coach scrub fuse 54 emits ([COACH_RATE_SHEET_mosaic_5min_fuse.md](COACH_RATE_SHEET_mosaic_5min_fuse.md))
- [ ] Full `match_4_full` P10 + P1 audited (`audit_batch_events_pack.py`)
- [ ] Full `match_4_full` P10 through **10+ min** scrubbed in Streamlit (`bash scripts/run_review_match4.sh data/output/match_4_full`)
- [ ] 3rd-match handover session

See [EVENT_ACCURACY_RERATE_2026-08-24.json](EVENT_ACCURACY_RERATE_2026-08-24.json) · [BATCH_EVENTS_AUDIT_20260825.md](BATCH_EVENTS_AUDIT_20260825.md)
