# Product Phase 1 scope

Current delivery focus for this repository. Broader roadmap: [PHASES.md](PHASES.md). Vision: [VISION.md](VISION.md).

## Constraints

| Constraint | Requirement |
|---|---|
| Target compute | RTX 5090-class GPU, Ubuntu 22.04 LTS |
| Camera ingest (planned) | Live streams via network switch using RTSP |
| Latency budget (live path) | Max ~200 ms algorithm run time |
| Data sources | Free for commercial use only; no licensed/proprietary match footage without permission |

Phase 1 delivery is **batch-first** (process match video files via `apps/batch_pipeline.py`). RTSP and the 200 ms budget shape design so Phase 2 live mode (`apps/live_pipeline.py`, `src/ingest/`) can reuse the same core.

## 1. Automated vision engine

- Aim for functional detection (~80% accuracy) to prove the pipeline.
- Prioritize **high precision over high recall** (correctness over completeness).
- **Confidence rule:** if tracking confidence drops below ~80% (e.g. blurry scramble), do nothing—only record events that meet the certainty bar.
- **Object tracking:** RF-DETR detection for players and ball; ByteTrack for multi-object tracking (not YOLO).
- **Team assignment:** classify Team A / B using color clustering.
- **Coordinate mapping:** pixel locations → pitch-relative `(x, y)`. Single center/master camera is expected to capture ~85% of action; multi-camera occlusion fixes are Phase 2.
- **Batch processing:** sequential processing of multiple match videos with operational reliability (checkpoints, incremental saves).

Messy or occluded frames may be skipped to preserve data quality. Perfecting those edge cases is Product Phase 2.

## 2. Heuristic event logic

- Physics / rule-based detection of: **pass**, **dribble**, **movement**, **recovery**, **shot**.
- Tune for wide-angle broadcast-style camera views.

## 3. Safety and review

- Incremental checkpoint writing so crashes do not lose progress.
- Streamlit dashboard to surface and review low-confidence flags.
- Manual corrections in the dashboard must update and persist to the core dataset.

## 4. Deliverables

- **Data export:** CSV/JSON with at least `Timestamp`, `Team_ID`, `Player_ID`, `Event`, `Location`.
- **Source code:** complete Python codebase for the pipeline.
- **App:** functional local Streamlit review application.

## 5. Acceptance criteria

1. Successfully process **2 full matches** through the pipeline.
2. Complete a **handover session** guiding the client through processing a **3rd match**.

## 6. Golden test set (player + ball)

**Match Gold100** is the canonical fixed benchmark for player and ball detection / classification on real Match 1 multi-cam frames.

- Spec + workflow: [GOLD100_PLAYER_BALL.md](GOLD100_PLAYER_BALL.md)
- Local pack: `data/processed/gold_sets/match1_1_100/` (rebuild via `scripts/gold_set/`; not stored in git)
- Correct labels: `python serve_viewer.py` → http://localhost:8080/gold100
- Eval: `python scripts/gold_set/eval_on_gold100.py --gold-dir data/processed/gold_sets/match1_1_100`

## Explicitly deferred to Product Phase 2+

- Multi-view fusion and occlusion elimination across cameras
- Predictive pre-event signaling and learned coaching models
- Kalman / short-horizon trajectory prediction for live cues
- Wearable command encoding and live haptic deployment
