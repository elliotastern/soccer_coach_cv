# Product phases

Naming note: **Product Phase 1/2/3** here are delivery milestones for the coaching product. They are separate from ball-detection *training* phases documented under [`docs/ball_detection/`](../ball_detection/).

## Product Phase 1 — Descriptive pipeline (current focus)

**Scope summary:** Python CV pipeline for tactical dataset generation (technical POC). See [PHASE1_SCOPE.md](PHASE1_SCOPE.md) for the four pillars — vision engine, heuristic events, safety/review, CSV/JSON export — and deliverables (source + Streamlit app).

Offline / batch-first vision and events:

- Detect and track players and ball (RF-DETR + ByteTrack)
- Team assignment via color clustering
- Pixel → pitch `(x, y)` mapping (single center/master camera)
- Heuristic event detection (pass, dribble, movement, recovery, shot)
- Checkpointed batch processing of match videos
- Streamlit review of low-confidence flags with corrections persisted
- CSV/JSON export of events and locations

Quality posture: **high precision, lower recall** — skip messy/occluded frames rather than guess. PoC **~80% accuracy** means **precision of published outputs ≥ 0.80** (emit only at conf ≥ ~0.80); full-field tiny-ball completeness is deferred. Details and acceptance: [PHASE1_SCOPE.md](PHASE1_SCOPE.md).

## Product Phase 2 — Predictive AI and sensor fusion

Shift from “what happened” to “what will happen / what to do”:

- Predictive models (RNN/Transformer-class) on Phase 1 structured state
- Pre-event signaling (cue pass/move/dribble before the action)
- Multi-view fusion: sync and merge up to 6 camera streams by timestamp
- Physics / Kalman smoothing for velocity and short-horizon position prediction
- Live RTSP processing on-premise with ~200 ms latency budget
- Encode 360° guidance commands for wearables (hex / PWM mixing)

## Product Phase 3 — Actionable coaching and deployment

- Integrate custom wearable hardware in a real training environment
- Continuous on-field haptic coaching during live sessions

## Out of scope for Product Phase 1

Do not treat these as current acceptance criteria:

- Perfecting occlusion, heavy blur, and edge-case accuracy (deferred to Phase 2)
- Multi-camera merge for full-field occlusion handling
- Predictive pre-event coaching models
- Live wearable PWM/hex protocol implementation and on-field deployment
- Replacing the heuristic event layer with learned event models (unless explicitly requested)

## Related docs

- [VISION.md](VISION.md) — Product goal and system concept
- [PHASE1_SCOPE.md](PHASE1_SCOPE.md) — Phase 1 requirements and acceptance
- [PITCH1_DIMENSIONS.md](PITCH1_DIMENSIONS.md) — Official Pitch 1 / Field 1 meters (not FIFA 105×68)
- [../architecture/LAYOUT.md](../architecture/LAYOUT.md) — Repository layout
- [../architecture/CONSTRAINTS.md](../architecture/CONSTRAINTS.md) — Latency / RTSP / data policy
