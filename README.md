# Soccer Analysis Pipeline

**GitHub Repository**: https://github.com/elliotastern/soccer_coach_cv

Automated football analysis pipeline using RF-DETR detection, ByteTrack tracking, and heuristic event detection.

## Architecture

Product-layer packages (see [docs/architecture/LAYOUT.md](docs/architecture/LAYOUT.md)):

- **perception** — Frame filtering, detection, tracking, team assignment
- **mapping** — Pixel → pitch `(x, y)` coordinates
- **events** — Heuristic event detection and checkpoints
- **review / export** — Streamlit review UI and CSV/JSON schemas
- **ingest** — File batch (Phase 1) and RTSP live adapter (Phase 2+)
- **coach / guidance / hardware** — Phase 2/3 stubs (predictive coach, haptics)

## Product scope

Long-term product vision (real-time AI coaching + haptics) and phased delivery:

- [docs/product/VISION.md](docs/product/VISION.md) — Goal, system components, value props
- [docs/product/PHASES.md](docs/product/PHASES.md) — Product Phase 1 / 2 / 3 roadmap
- [docs/product/PHASE1_SCOPE.md](docs/product/PHASE1_SCOPE.md) — Current Phase 1 requirements and acceptance
- [docs/architecture/CONSTRAINTS.md](docs/architecture/CONSTRAINTS.md) — Latency, RTSP, data policy
- [docs/architecture/LAYOUT.md](docs/architecture/LAYOUT.md) — Repository layout

Current repo focus is **Product Phase 1** (batch vision + heuristic events). Product Phase naming is separate from ball-detection training phases under `docs/ball_detection/`.

## Precision & Quantization Strategy

**IMPORTANT:** This project uses a specific precision strategy optimized for tiny object detection (<15 pixels):

### 1. Training: Mixed Precision (FP16/FP32)
- **Status:** ✅ Active (RF-DETR default `amp=True`)
- **Why:** Essential to capture tiny gradients of small objects
- **Result:** ~2x faster training with minimal accuracy loss

### 2. MVP Deployment: FP16 (Half Precision)
- **Status:** ✅ Active for all devices (CUDA and CPU)
- **Why:** Safest start. ~3x speedup on NVIDIA GPUs with zero accuracy loss
- **Implementation:** `model.half()` in `src/perception/local_detector.py`
- **Use this for:** First production release

### 3. Future Optimization: INT8 via QAT (Quantization-Aware Training)
- **Status:** 🔄 Future optimization only (if FP16 is too slow)
- **Critical:** Use **QAT** (Quantization-Aware Training), **NOT PTQ** (Post-Training Quantization)
- **Why:** QAT preserves tiny object detection; PTQ may lose it
- **When:** Edge devices, mobile, very slow inference requirements

**Key Rule:** For tiny object detection, always use QAT for INT8, never PTQ.

See [docs/architecture/DEPLOYMENT_STRATEGY.md](docs/architecture/DEPLOYMENT_STRATEGY.md) for detailed implementation guide.


## Setup

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

**Key Dependencies:**
- **RF-DETR** ([roboflow/rf-detr](https://github.com/roboflow/rf-detr)) – detection model (`pip install rfdetr`)
- PyTorch & Torchvision (training)
- Transformers (DETR model)
- MLflow (experiment tracking)
- TensorBoard (metrics visualization)
- Streamlit (dashboard)

### 2. Configure Environment

Create a `.env` file with your Roboflow API key:

```bash
ROBOFLOW_API_KEY=your_api_key_here
```

### 3. Configure Model

Edit `configs/default.yaml` and set your RF-DETR model ID:

```yaml
roboflow:
  model_id: "your-model-id"
```

## Usage

### Process Video

```bash
python apps/batch_pipeline.py --video path/to/video.mp4 --config configs/default.yaml --output data/processed
# or: python main.py ...  (thin wrapper)
```

### Player detection on 37a (20 frames)

Uses the person model `models/checkpoint_best_total_after_100_epochs.pth` and video `data/raw/37CAE053-841F-4851-956E-CBF17A51C506.mp4`. Output: `data/output/37a_20frames/`.

```bash
./run_player_detection_37a_20frames.sh
```

Or manually:

```bash
python scripts/process_video_pipeline.py \
  "data/raw/37CAE053-841F-4851-956E-CBF17A51C506.mp4" \
  --model models/checkpoint_best_total_after_100_epochs.pth \
  --output data/output/37a_20frames \
  --max-frames 20
```

### Review Dashboard

```bash
streamlit run apps/review_dashboard.py
```

Or use the RunPod script:

```bash
./runpod.sh
```

## Output

The pipeline generates:

- `events.json`: Event-centric JSON output
- `events.csv`: Frame-by-frame CSV data
- `frame_data.csv`: Detailed frame data
- `checkpoints/`: Periodic checkpoint files

## Configuration

### Default Config (`configs/default.yaml`)

- Detection thresholds
- Tracker parameters (ByteTrack)
- Event detection thresholds
- Checkpoint intervals

### Zones Config (`configs/zones.yaml`)

Defines tactical zones (Zone 14, Half-Spaces, Goal Area, etc.)

## Ball model and training

Ball-specific training, resume configs, and validation utilities live in the same repository.

- **Train / resume / finetune**: `python scripts/train_ball.py` with `configs/training.yaml`, `configs/training_finetune.yaml`, or `configs/resume_*.yaml` (see [configs/RESUME_CONFIGS_README.md](configs/RESUME_CONFIGS_README.md)).
- **100-frame validation (HTML)**: scripts in [scripts/ball_validation/](scripts/ball_validation/).
- **Analysis variants (first N frames)**: [scripts/experiments/](scripts/experiments/).
- **Operator notes**: [docs/ball_detection/](docs/ball_detection/) (training reports, MLflow notes, strategies).
- **Hugging Face model card source** (YAML frontmatter for Hub): [docs/huggingface_model_card.md](docs/huggingface_model_card.md).
- **Upload to Hub** (optional): `python scripts/push_to_huggingface.py --repo-id <your-org>/<model-repo>` (defaults to this repo root; does not overwrite the root `README.md`).
- **COCO dataset archive (SCP / extract)**: [docs/runbooks/DOWNLOAD_INSTRUCTIONS.md](docs/runbooks/DOWNLOAD_INSTRUCTIONS.md).

Published Hub example: [eeeeeeeeeeeeee3/soccer-ball-detection](https://huggingface.co/eeeeeeeeeeeeee3/soccer-ball-detection).

## Docker

Build and run:

```bash
docker build -t soccer-analysis .
docker run -p 8501:8501 soccer-analysis
```

## Project Structure

Full tree: [docs/architecture/LAYOUT.md](docs/architecture/LAYOUT.md).

```
soccer_coach_cv/
├── apps/
│   ├── batch_pipeline.py      # Phase 1 match processing
│   ├── live_pipeline.py       # Phase 2+ RTSP stub
│   └── review_dashboard.py    # Streamlit review
├── main.py                    # Thin wrapper → apps/batch_pipeline.py
├── configs/
├── src/
│   ├── perception/            # Detect, track, team assign
│   ├── mapping/               # Pixel → pitch
│   ├── events/                # Heuristic events
│   ├── state/                 # Shared types
│   ├── export/                # CSV/JSON schemas
│   ├── ingest/                # File + RTSP adapters
│   ├── review/                # Streamlit UI
│   ├── coach/                 # Phase 2+ stubs
│   ├── guidance/              # Phase 2+ haptic stubs
│   ├── training/
│   ├── analysis/              # Compatibility shims
│   └── visualization/         # Compatibility shims → review
├── scripts/                   # CLIs, experiments, training helpers
├── docs/
│   ├── product/
│   ├── architecture/
│   ├── runbooks/
│   └── ball_detection/
├── annotation/
├── data/
│   ├── raw/
│   ├── interim/
│   ├── processed/             # Preferred outputs
│   ├── external/
│   └── output -> processed    # Legacy alias
├── models/
├── notebooks/
├── reports/
├── tests/{unit,integration,latency}/
└── hardware/                  # Phase 3 wearable notes
```

Editor-specific secrets (tokens, machine paths) stay **local** only. Optional team guardrails may live under `.cursor/rules/` (no credentials).

## Event Types

- **Pass**: Ball movement with velocity > threshold
- **Dribble**: Player maintains close control of ball
- **Shot**: High-velocity ball movement toward goal
- **Recovery**: Player gains possession of ball
- **Movement**: General player movement

## Legal & Technical Guardrails

- Only synthetic data (SoccerSynth-Detection dataset)
- No licensed/proprietary match footage
- RF-DETR for detection (not YOLO)
- COCO JSON format for detections
- ByteTrack for multi-object tracking
- **Precision Strategy:** Mixed Precision (FP16/FP32) for training, FP16 for MVP deployment, QAT (not PTQ) for future INT8 optimization
