# CVAT Auto-Ingest Watchdog

Automated pipeline for processing soccer videos with RF-DETR, detecting events, and uploading to CVAT.

## Overview

The `auto_ingest.py` script monitors a directory for new video files, runs inference with your trained RF-DETR model, detects events (passes, shots, dribbles, recoveries), generates CVAT-compatible XML annotations, and automatically uploads everything to CVAT.

## Setup

### 1. Install Dependencies

```bash
pip install -r requirements_auto.txt
```

### 2. Configure Environment

Copy the environment template and fill in your values:

```bash
cp env.template .env
# Edit .env with your CVAT credentials and model path
```

Required environment variables:
- `CVAT_URL`: CVAT server URL (default: http://localhost:8080)
- `CVAT_USER`: CVAT username (default: admin)
- `CVAT_PASS`: CVAT password (default: admin)
- `MODEL_PATH`: Path to your trained RF-DETR checkpoint

### 3. Configure Settings

Edit `configs/auto_ingest.yaml` to adjust:
- Watch directory path
- Model checkpoint path
- Detection/tracking thresholds
- Event detection parameters
- CVAT connection settings

## Usage

### Start the Watchdog

```bash
python auto_ingest.py
```

The script will:
1. Monitor `./incoming_videos/` (or path in config) for new `.mp4`, `.avi`, `.mov`, `.mkv` files
2. Wait for files to be fully written (size stability check)
3. Run RF-DETR inference on each frame
4. Track objects using ByteTrack
5. Detect events (passes, shots, dribbles, recoveries)
6. Generate CVAT XML 1.1 format with bounding boxes and event tags
7. Create a CVAT task and upload video + annotations
8. Move processed files to `./processed/` directory

### Workflow

1. **Drop video files** into the watch directory (e.g., `./incoming_videos/match_01.mp4`)
2. **Wait for processing** - the script logs progress to console and `auto_ingest.log`
3. **Check CVAT** - log into CVAT at the configured URL to see the new task with annotations
4. **Review and correct** - use CVAT's interface to review low-confidence detections and correct annotations
5. **Export** - export corrected annotations from CVAT in your preferred format (COCO, YOLO, etc.)

## Output Format

### CVAT XML Structure

The generated XML includes:

- **Tracks**: Bounding boxes for players and ball with track IDs
  - Each track has `<box>` elements for each frame
  - Confidence scores stored as attributes
  
- **Event Tags**: Timeline markers for detected events
  - Single-frame events (shot, recovery): `<tag>` at specific frame
  - Duration events (pass, dribble): `<tag>` at start frame
  - Confidence and involved players stored as attributes

### Example XML Structure

```xml
<annotations>
  <version>1.1</version>
  <track id="0" label="player" source="auto">
    <box frame="0" xtl="100" ytl="200" xbr="150" ybr="300" ...>
      <attribute name="confidence">0.85</attribute>
    </box>
  </track>
  <tag label="shot" frame="500" source="auto">
    <attribute name="confidence">0.92</attribute>
    <attribute name="involved_players">5</attribute>
  </tag>
</annotations>
```

## Features

- **File Locking**: Waits for files to be fully written before processing
- **Duplicate Prevention**: Tracks processed files to avoid reprocessing
- **Error Handling**: Graceful error handling with detailed logging
- **Progress Tracking**: Logs frame-by-frame progress during inference
- **Event Support**: Detects and annotates soccer events (pass, shot, dribble, recovery)
- **Confidence Scoring**: Stores model confidence for filtering in CVAT

## Troubleshooting

### CVAT Connection Issues

- Verify CVAT is running: `curl http://localhost:8080/api/server/about`
- Check credentials in `.env` file
- Ensure CVAT SDK is installed: `pip install cvat-sdk`

### Model Loading Issues

- Verify model checkpoint path in config
- Check that model was trained with compatible architecture
- Ensure CUDA is available if `use_cuda: true`

### File Processing Issues

- Check `auto_ingest.log` for detailed error messages
- Verify video file is not corrupted
- Ensure sufficient disk space in processed directory

## Integration with Training Pipeline

After correcting annotations in CVAT:

1. Export annotations from CVAT (COCO or YOLO format)
2. Use exported data to retrain/fine-tune your RF-DETR model
3. Update `MODEL_PATH` in config to use the new model
4. The improved model will be used for future auto-ingest runs

## Configuration Reference

See `configs/auto_ingest.yaml` for all configuration options:

- **paths**: Watch and processed directories
- **model**: Model checkpoint path and CUDA settings
- **detection**: Confidence thresholds
- **tracker**: ByteTrack parameters
- **mapping**: Pitch coordinate mapping
- **events**: Event detection thresholds
- **cvat**: CVAT connection settings
