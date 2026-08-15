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

- Prioritize **high precision over high recall** (correctness over completeness).
- **Object tracking:** RF-DETR detection for players and ball; ByteTrack for multi-object tracking (not YOLO).
- **Team assignment:** classify Team A / B using color clustering.
- **Coordinate mapping:** pixel locations → pitch-relative `(x, y)`. Single center/master camera is expected to capture ~85% of action; multi-camera occlusion fixes are Phase 2.
- **Batch processing:** sequential processing of multiple match videos with operational reliability (checkpoints, incremental saves).

### PoC accuracy definition (~80%)

Client / proof-of-concept **“approximately 80% accuracy”** means the following (not all-frame recall at high confidence):

| Role | Metric | Target |
|---|---|---|
| **Primary (acceptance)** | Precision of **emitted** ball (and later event) outputs at IoU 0.5 | **≥ 0.80** |
| **Secondary (coverage)** | **Clear-ball recall** — ball on a near/clear view (tunable; start with min box side ≥ 25 px on full-res 4K, not heavily occluded) | **≥ 0.80** |
| **Not the PoC bar** | Per-frame Match/Gold recall @ conf ≥ 0.8 on all frames (including tiny far balls) | Engineering stretch / Phase 2+ completeness |

- **Emit / confidence rule:** if confidence drops below ~0.80 (e.g. blurry scramble), **do nothing**—only publish detections and events that meet the certainty bar.
- The raw detector may feed **lower** scores into ByteTrack for association; the **product emit gate** remains ≥ ~0.80 (per detection or tracklet EMA) so published precision holds.
- Measure primary precision on [Match Gold100](GOLD100_PLAYER_BALL.md) (and/or an agreed demo clip). Report recall and clear-ball recall separately; they are not a substitute for P_emit.
- **Checkpoint pick during ball finetune:** use Gold strip 0–49 PoC (`P_emit`, clear-ball R), not train-pack AP50. See [TRAIN_CHECKPOINT_SELECTION.md](../ball_detection/TRAIN_CHECKPOINT_SELECTION.md).

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

### 5.1 Detection metrics (PoC)

1. **Ball PoC pass:** precision of emitted ball predictions **P_emit ≥ 0.80** at IoU 0.5, conf ≥ 0.80, on Match Gold100 and/or an agreed demo clip.
2. **Clear-ball recall** reported separately (secondary coverage target ≥ 0.80); do **not** fail Phase 1 solely on all-frame recall @ conf ≥ 0.80.
3. Player detection follows the same precision-first emit gate; Gold100 remains the fixed benchmark ([GOLD100_PLAYER_BALL.md](GOLD100_PLAYER_BALL.md)).

## 6. Golden test set (player + ball)

**Match Gold100** is the canonical fixed benchmark for player and ball detection / classification on real Match 1 multi-cam frames.

- Spec + workflow: [GOLD100_PLAYER_BALL.md](GOLD100_PLAYER_BALL.md)
- Local pack: `data/processed/gold_sets/match1_1_100/` (rebuild via `scripts/gold_set/`; not stored in git)
- Correct labels: `python serve_viewer.py` → http://localhost:8080/gold100
- Eval: `python scripts/gold_set/eval_on_gold100.py --gold-dir data/processed/gold_sets/match1_1_100`
- Ball PoC / rank: `python scripts/gold_set/eval_poc_ball_metrics.py --strip-max 49 --require-ball-gt` and [TRAIN_CHECKPOINT_SELECTION.md](../ball_detection/TRAIN_CHECKPOINT_SELECTION.md)

**Match2 Top Left 300-frame gold labels** — dense Match 2 P10 clip (0:26–0:31, 300 @ 60 fps) for the 4-quad Top Left window; does not replace Gold100. Spec: [MATCH2_4QUAD_TOP_LEFT_300.md](MATCH2_4QUAD_TOP_LEFT_300.md).

### 6.1 Six P-cam ball system goal (Match 2)

Client-ok path: combine **P1, P6, P7, P8, P10, P12** for system ball **R ≥ 0.80** and **P ≥ 0.90**, with a live ball-path budget of **≤ 125 ms** on RTX 5090 (stricter than the 200 ms e2e doc budget). Start with max_conf / soft 2-cam co-occurrence (no dense SAHI); **epipolar consensus is gated on Match 2 calib** (not on disk yet).

- Baseline + consensus eval: `python3 scripts/gold_set/eval_match2_top_left_multicam_baseline.py`
- Reports: `reports/eval_match2_v10/top_left_multicam_baseline/`, `reports/eval_match2_v10/top_left_multicam_consensus/`

## Explicitly deferred to Product Phase 2+

- Full multi-view **fusion / occlusion merge** (beyond best-cam / soft consensus)
- Predictive pre-event signaling and learned coaching models
- Kalman / short-horizon trajectory prediction for live cues
- Wearable command encoding and live haptic deployment
