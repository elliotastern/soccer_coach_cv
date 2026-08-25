# Coach scrub checklist — events (Phase 1)

Use while gates are green but **fuse vs single-cam** mismatches need human truth.

## Before scrubbing

1. Pull latest batch: `bash scripts/pull_match4_full_from_catch.sh`
2. Streamlit Expert: `bash scripts/run_review_match4.sh data/output/match_4_5min` (or `match_4_full`)
3. Rate sheets (fill while scrubbing):
   - [COACH_RATE_SHEET_match_4_5min__P10-match4.md](COACH_RATE_SHEET_match_4_5min__P10-match4.md) — **16 pass**
   - [COACH_RATE_SHEET_match_4_5min__P8-match4.md](COACH_RATE_SHEET_match_4_5min__P8-match4.md) — **18 pass**
4. When staged: `bash scripts/pull_from_catch.sh` for coach mosaic MP4s
5. Auto-wait: `bash scripts/catch_wait_mosaic_pull.sh` (pulls when mosaic + eval clips ready)

## Holdout windows (fuse truth — not primary gate)

| Clip | Match time | Batch single-cam | Coach action |
|------|------------|------------------|--------------|
| `real_fuse_eval_49s` | ~49–64 s | 3 pass | Watch stride-4 MP4 · confirm movement/dribble vs pass |
| `real_fuse_eval_69s` | ~70–85 s | pass cluster | Same — verify fuse before locking labels |
| `real_fuse_eval_20s` | ~20–35 s | pass cluster | movement labeled — reference |

HTML: [real_fuse_eval_49s_review.html](real_fuse_eval_49s_review.html)

## 10-minute Streamlit scrub (5-min quad)

| Time block | What to rate |
|------------|----------------|
| 0–2 min | Pass emits — precision (real pass vs noise) |
| 2–4 min | Sparse events — miss vs correct silence |
| 4–5 min | Dribble — any at conf≥0.8? (expect rare on single-cam) |

Log notes in handover UI; do **not** change emit conf to “fix” sparse dribble.

## After scrub

- Lock `labels.json` for holdouts → re-score: `python3 scripts/eng_loop_heuristic_events.py`
- If fuse pattern repeats (e.g. movement not pass): document in `DEAD_ENDS.md` before rule change
- Primary gate stays `real_fuse_15s` stride 4 — do not swap holdout into primary

## Catch automation

```bash
bash scripts/catch_queue_fuse_eval_mosaics.sh   # 49s + 69s MP4s after mosaic5
bash scripts/catch_poll_pull.sh               # auto-pull growing full batch
```
