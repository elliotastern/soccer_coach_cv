# Soccer Coach AI

Modular Production Architecture for soccer video analysis using RF-DETR and computer vision.

## Project Structure

```
soccer-ai/
├── .cursorrules           # Your legal & tech guardrails
├── requirements.txt       # The business-safe libraries
├── main.py                # Entry point: The video loop
├── configs/               # Hyperparameters (thresholds, colors)
│   └── default.yaml
├── data/
│   ├── raw/               # Original mp4 files
│   └── output/            # Videos with boxes/JSON results
├── src/                   # Proprietary Source Code
│   ├── __init__.py
│   ├── detector.py        # RF-DETR wrapper
│   ├── tracker.py         # ByteTrack & Supervision logic
│   ├── logic/             # THE BUSINESS BRAIN
│   │   ├── team_id.py     # K-Means clustering code
│   │   └── mapping.py     # Pitch homography math
│   └── utils/             # Visualization & video helpers
├── models/                # Where .pt or .engine files live
└── tests/                 # Benchmark scripts
```

## Directory Descriptions

### Core Files
- **main.py**: Main entry point that processes video files through the detection and tracking pipeline
- **requirements.txt**: Python dependencies (Apache 2.0 / MIT licensed packages)
- **.cursorrules**: Development guidelines and coding standards

### Configuration
- **configs/**: YAML configuration files for hyperparameters, detection thresholds, color schemes, and other settings

### Data Management
- **data/raw/**: Input video files (MP4 format)
- **data/output/**: Processed videos with bounding boxes and JSON results containing detection/tracking data

### Source Code (`src/`)
- **detector.py**: RF-DETR model wrapper for object detection
- **tracker.py**: ByteTrack and Supervision integration for multi-object tracking
- **logic/team_id.py**: K-Means clustering algorithm for team identification
- **logic/mapping.py**: Pitch homography transformations for coordinate mapping
- **utils/**: Helper functions for visualization, video processing, and utilities

### Models
- **models/**: Directory for trained model files (`.pt` PyTorch models or `.engine` TensorRT optimized models)

### Testing
- **tests/**: Benchmark scripts and test cases for validation

## Setup

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Place your model files in the `models/` directory

3. Add raw video files to `data/raw/`

4. Run the main script:
```bash
python main.py
```

## License

This project uses Apache 2.0 and MIT licensed dependencies. See `requirements.txt` for details.
