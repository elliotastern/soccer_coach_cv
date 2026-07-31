# Product vision

## Goal

Build an automated, real-time AI football (soccer) coaching system that delivers intelligent haptic guidance directly to players on the pitch—gamified training that bridges video-game style coaching and live practice.

This repository currently implements the **offline/batch computer-vision and heuristic-events foundation** toward that product (Product Phase 1). Predictive coaching and wearable deployment are later phases.

## System architecture

Three integrated components:

1. **AI computer vision network** — Field cameras continuously track players and the ball to produce pitch-relative `(x, y)` coordinates.
2. **Coach Box (tactical brain)** — On-field, low-latency processing that analyzes positioning, team shape, space, and game situations.
3. **Smart haptic wearables** — Multi-zone bands that turn AI decisions into vibration cues:
   - **Wrist bands:** Action signals (pass ground vs lofted, shoot, dribble, recover).
   - **Shin bands:** 360° directional movement guidance (4-motor PWM mixing) for run, cut, press, or spacing.

## Value proposition

- **Accelerated player development** — Real-time positioning and tactical cues for developing players.
- **Gamified real-world training** — Structured, interactive drills between video games and live coaching.
- **Scalable professional coaching** — Academy/club deployment without relying only on sideline staff availability.
- **Proprietary data engine** — Time-series match state for continuous improvement of predictive models.

## Detection stack (this repo)

Use **RF-DETR** for detection and **ByteTrack** for tracking. Early product briefs mentioned YOLO; this codebase standardizes on RF-DETR (not YOLO).

## Related docs

- [PHASES.md](PHASES.md) — Product Phase 1 / 2 / 3 roadmap
- [PHASE1_SCOPE.md](PHASE1_SCOPE.md) — Current delivery requirements and acceptance
- [../architecture/LAYOUT.md](../architecture/LAYOUT.md) — Repository layout
- [../architecture/CONSTRAINTS.md](../architecture/CONSTRAINTS.md) — Runtime and data constraints
