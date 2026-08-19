# Repository layout

Target layout for the Soccer Coach CV product (cookiecutter-style data dirs + product-layer packages).

```text
soccer_coach_cv/
├── apps/                      # Entrypoints
│   ├── batch_pipeline.py      # Product Phase 1 match processing
│   ├── live_pipeline.py       # Phase 2+ RTSP stub
│   └── review_dashboard.py    # Streamlit review
├── configs/
├── data/
│   ├── raw/                   # Input videos / captures
│   ├── interim/               # Checkpoints / partial results
│   ├── processed/             # Tracks, events, pitch coords
│   └── external/              # Third-party commercial-safe datasets
├── docs/
│   ├── product/               # Vision, phases, Phase 1 scope, Pitch 1 meters, Match 2/3 capture
│   ├── architecture/          # Constraints, deployment, mapping plans
│   ├── runbooks/              # Operator / training guides
│   └── ball_detection/        # Ball training notes (not product phases)
├── hardware/                  # Phase 3 wearable notes / protocol
├── models/                    # Checkpoints
├── notebooks/                 # Exploration only
├── reports/                   # Eval HTML / acceptance evidence
├── scripts/                   # One-off CLIs and experiments
│   └── gold_set/              # Match Gold100 build / review / eval (player+ball)
├── src/
│   ├── perception/            # Detect, track, team assign
│   ├── mapping/               # Pixel → pitch (x, y)
│   ├── events/                # Heuristic events
│   ├── state/                 # Shared types
│   ├── export/                # CSV/JSON schemas
│   ├── ingest/                # File + RTSP adapters
│   ├── review/                # Streamlit review UI
│   ├── coach/                 # Phase 2+ tactical brain stubs
│   ├── guidance/              # Phase 2+ haptic encoding stubs
│   ├── training/              # Model training utilities
│   ├── analysis/              # Compatibility shims → mapping/events/coach
│   └── visualization/         # Compatibility shims → review
├── tests/
│   ├── unit/
│   ├── integration/
│   └── latency/
└── annotation/                # CVAT / COCO utilities
```

Legacy imports (`src.analysis.*`, `src.types`, `src.visualization.*`) still work via shims. Prefer the new package paths in new code.

Official pitch meters: [product/PITCH1_DIMENSIONS.md](../product/PITCH1_DIMENSIONS.md) (Pitch 1 / Field 1, not FIFA 105×68).
