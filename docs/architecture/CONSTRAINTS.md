# Runtime and data constraints

Target environment and hard limits for the coaching product. See also [product/PHASE1_SCOPE.md](../product/PHASE1_SCOPE.md).

## Compute and OS

- Target GPU: RTX 5090-class
- Target OS: Ubuntu 22.04 LTS

## Ingest and latency

- Planned live ingest: RTSP from PoE cameras via network switch
- Live algorithm budget: **≤ 200 ms** end-to-end
- Product Phase 1 delivery is batch/file-first (`apps/batch_pipeline.py`); live path is reserved at `apps/live_pipeline.py` / `src/ingest/`

## Data policy

- Free for commercial use (or explicitly licensed) only
- No licensed/proprietary match footage without permission
- Prefer synthetic datasets (e.g. SoccerSynth-Detection) for training and public examples
- **Do not use** `football.v1i` (Roboflow `football-r4il1`) — league/broadcast footage; blocked in the shared data catalog even if listed as CC BY

## Quality posture (Phase 1)

- High precision over high recall
- Tracking confidence below ~80% → drop / do nothing

## Pitch

Official field is **Pitch 1 / Field 1**: [product/PITCH1_DIMENSIONS.md](../product/PITCH1_DIMENSIONS.md) (**53.90 × ~34.8 m**). Do not use FIFA 105×68.
