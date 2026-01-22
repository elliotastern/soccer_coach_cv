# Soccer Analysis Pipeline

**GitHub Repository**: https://github.com/elliotastern/soccer_coach_cv

Automated football analysis pipeline using RF-DETR detection, ByteTrack tracking, and heuristic event detection.

## Architecture

The pipeline follows a modular architecture:

- **Perception Layer**: Frame filtering, detection, tracking, team assignment
- **Analysis Layer**: Coordinate mapping, event detection, event aggregation
- **Visualization Layer**: Review dashboard, annotation tools

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

See `docs/DEPLOYMENT_STRATEGY.md` for detailed implementation guide.

## Setup

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

**Key Dependencies:**
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
python main.py --video path/to/video.mp4 --config configs/default.yaml --output data/output
```

### Review Dashboard

```bash
streamlit run src/visualization/app.py
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

## Docker

Build and run:

```bash
docker build -t soccer-analysis .
docker run -p 8501:8501 soccer-analysis
```

## Project Structure

```
soccer_coach_cv/
├── main.py                 # Main orchestrator
├── configs/               # Configuration files
├── src/
│   ├── perception/        # Detection, tracking, team assignment
│   ├── analysis/         # Mapping, event detection
│   ├── visualization/    # Streamlit dashboard
│   ├── types.py          # Data classes
│   └── schema.py        # Output schemas
├── data/
│   ├── raw/              # Input videos
│   └── output/           # Generated outputs
└── models/               # Model files

```

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
