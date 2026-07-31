---
license: mit
tags:
- computer-vision
- object-detection
- soccer
- ball-detection
- detr
- rf-detr
- pytorch
datasets:
- custom
metrics:
- mAP
- precision
- recall
---

# Soccer Ball Detection with RF-DETR

Automated soccer ball detection pipeline using RF-DETR (Roboflow DETR) optimized for tiny object detection (<15 pixels).

## Model Details

### Architecture

- **Model**: RF-DETR Base
- **Backbone**: ResNet-50
- **Classes**: Ball (single class detection)
- **Input Resolution**: 1120x1120 (optimized for memory)
- **Precision**: Mixed Precision (FP16/FP32) training, FP16 inference

### Performance

Based on training evaluation report (Epoch 39):

- **mAP@0.5:0.95**: 0.682 (68.2%)
- **mAP@0.5**: 0.990 (99.0%)
- **Small Objects mAP**: 0.598 (59.8%)
- **Training Loss**: 3.073
- **Validation Loss**: 3.658

## Quick Start

### Installation

```bash
pip install -r requirements.txt
```

### Usage

`LocalDetector` loads a PyTorch checkpoint (see `src/perception/local_detector.py`). Pass a **BGR `numpy` frame** (OpenCV), not a file path.

```python
import cv2
from src.perception.local_detector import LocalDetector

detector = LocalDetector(
    model_path="models/checkpoints/latest_checkpoint.pth",
    confidence_threshold=0.5,
)
frame_bgr = cv2.imread("path/to/frame.jpg")
results = detector.detect(frame_bgr)
```

### Process Video

```bash
python apps/batch_pipeline.py --video path/to/video.mp4 --config configs/default.yaml --output data/processed
```

## Training

### Train from Scratch

```bash
python scripts/train_ball.py \
    --config configs/training.yaml \
    --dataset-dir datasets/combined \
    --output-dir models \
    --epochs 50
```

### Resume Training

```bash
python scripts/train_ball.py \
    --config configs/resume_20_epochs.yaml \
    --dataset-dir datasets/combined \
    --output-dir models \
    --resume models/checkpoints/latest_checkpoint.pth \
    --epochs 50
```

## Configuration

Key configuration files:

- `configs/training.yaml` — Main training configuration
- `configs/training_finetune.yaml` — Lower LR / batch finetune from a checkpoint
- `configs/default.yaml` — Roboflow / pipeline inference defaults (separate from `LocalDetector` checkpoint loading)
- `configs/resume_*.yaml` — Resume training configurations

## Datasets

This model was trained on:

- SoccerSynth-Detection (synthetic data)
- Open Soccer Ball Dataset
- Custom validation sets

## Precision Strategy

- **Training**: Mixed Precision (FP16/FP32) — RF-DETR default
- **Inference**: FP16 (half precision) for ~3x speedup
- **Future**: INT8 via QAT (Quantization-Aware Training) for edge devices

## Citation

```bibtex
@software{soccer_ball_detection,
  title={Soccer Ball Detection with RF-DETR},
  author={Your Name},
  year={2026},
  url = {https://huggingface.co/eeeeeeeeeeeeee3/soccer-ball-detection}
}
```

## License

MIT License — See LICENSE file for details.

## Acknowledgments

- RF-DETR by Roboflow
- SoccerSynth-Detection dataset
- Open Soccer Ball Dataset
