"""Local dual RF-DETR detector: people + ball checkpoints."""
from pathlib import Path
from typing import List, Optional

import cv2
import numpy as np
from PIL import Image

from src.state.types import Detection

_RFDETR_IMPORT_PATCHED = False


def _patch_rfdetr_imports() -> None:
    """Avoid peft→tensorflow hang on macOS when importing rfdetr."""
    global _RFDETR_IMPORT_PATCHED
    if _RFDETR_IMPORT_PATCHED:
        return
    import importlib.machinery
    import os
    import sys
    import types

    os.environ.setdefault("USE_TF", "0")
    os.environ.setdefault("TRANSFORMERS_NO_TF", "1")
    os.environ.setdefault("USE_TORCH", "1")

    def stub(name: str, **attrs):
        module = types.ModuleType(name)
        module.__spec__ = importlib.machinery.ModuleSpec(name, loader=None)
        module.__file__ = f"<stub:{name}>"
        for key, value in attrs.items():
            setattr(module, key, value)
        sys.modules[name] = module
        return module

    if "tensorflow" not in sys.modules:
        class _GFile:
            @staticmethod
            def join(*parts):
                return os.path.join(*parts)

        class _IO:
            gfile = _GFile()

        stub("tensorflow", io=_IO())
        stub("tensorflow_probability")
        stub("tensorflow_text")

    if "torch.utils.tensorboard" not in sys.modules:
        class SummaryWriter:
            def __init__(self, *args, **kwargs):
                pass

            def add_scalar(self, *args, **kwargs):
                pass

            def close(self):
                pass

        tb = stub("torch.utils.tensorboard", SummaryWriter=SummaryWriter, FileWriter=SummaryWriter)
        stub("torch.utils.tensorboard.writer", SummaryWriter=SummaryWriter, FileWriter=SummaryWriter)
        stub("torch.utils.tensorboard._embedding")
        _ = tb

    _RFDETR_IMPORT_PATCHED = True


def _require_checkpoint(path: str, label: str) -> Path:
    checkpoint = Path(path)
    if not checkpoint.is_file():
        raise FileNotFoundError(f"{label} checkpoint not found: {checkpoint}")
    return checkpoint


def _xyxy_to_xywh(bbox_xyxy) -> Optional[tuple]:
    x_min, y_min, x_max, y_max = map(float, bbox_xyxy)
    width = x_max - x_min
    height = y_max - y_min
    if width <= 0 or height <= 0:
        return None
    return (x_min, y_min, width, height)


def _frame_to_pil(frame: np.ndarray) -> Image.Image:
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    return Image.fromarray(frame_rgb)


def _parse_rfdetr_detections(detections_raw, class_id: int, class_name: str) -> List[Detection]:
    results = []
    if not hasattr(detections_raw, "class_id"):
        return results
    for i in range(len(detections_raw.class_id)):
        bbox = _xyxy_to_xywh(detections_raw.xyxy[i])
        if bbox is None:
            continue
        results.append(Detection(
            class_id=class_id,
            confidence=float(detections_raw.confidence[i]),
            bbox=bbox,
            class_name=class_name,
        ))
    return results


def load_people_model(checkpoint_path: str):
    _patch_rfdetr_imports()
    from rfdetr import RFDETRMedium

    path = _require_checkpoint(checkpoint_path, "People")
    print(f"Loading people RF-DETR from: {path}")
    model = RFDETRMedium(pretrain_weights=str(path))
    print("✅ People model loaded")
    return model


def load_ball_model(checkpoint_path: str):
    _patch_rfdetr_imports()
    from rfdetr import RFDETRBase

    path = _require_checkpoint(checkpoint_path, "Ball")
    print(f"Loading ball RF-DETR from: {path}")
    # RFDETRBase(pretrain_weights=...) reads class_names from checkpoint args
    # (ball_89.pth has class_names=['ball'], num_classes=2 = len(classes)+1)
    model = RFDETRBase(pretrain_weights=str(path))
    print("✅ Ball model loaded")
    return model


class LocalRFDETRDetector:
    """People + ball RF-DETR with the same detect() interface as Detector."""

    def __init__(
        self,
        player_checkpoint: str,
        ball_checkpoint: str,
        confidence_threshold: float = 0.5,
        player_class_id: int = 0,
        ball_class_id: int = 1,
        enhance_ball: bool = False,
        use_sahi: bool = False,
        use_kalman: bool = False,
    ):
        self.confidence_threshold = confidence_threshold
        self.player_class_id = player_class_id
        self.ball_class_id = ball_class_id
        self.enhance_ball = enhance_ball
        self.use_sahi = use_sahi
        self.use_kalman = use_kalman
        self.people_model = load_people_model(player_checkpoint)
        self.ball_model = load_ball_model(ball_checkpoint)
        self._ball_prelabeler = None
        if enhance_ball or use_sahi or use_kalman:
            from src.perception.ball_prelabel import BallPrelabelConfig, BallPrelabeler

            # Match stack: size filter + optional SAHI recover when fullframe empty
            self._ball_prelabeler = BallPrelabeler(
                self.ball_model,
                BallPrelabelConfig(
                    threshold=min(0.30, confidence_threshold)
                    if confidence_threshold > 0
                    else 0.30,
                    use_sahi=use_sahi,
                    sahi_fallback_only=True,
                    sahi_recover_only=True,
                    use_size_filter=True,
                    topk=2,
                    use_kalman=use_kalman,
                    min_side=4,
                    max_side=240,
                ),
                class_id=ball_class_id,
            )

    def detect(self, frame: np.ndarray) -> List[Detection]:
        pil_image = _frame_to_pil(frame)
        threshold = self.confidence_threshold

        people_raw = self.people_model.predict(pil_image, threshold=threshold)
        detections = _parse_rfdetr_detections(
            people_raw, self.player_class_id, "player"
        )
        if self._ball_prelabeler is not None:
            detections.extend(self._ball_prelabeler.detect_pil(pil_image))
        else:
            ball_raw = self.ball_model.predict(pil_image, threshold=threshold)
            detections.extend(_parse_rfdetr_detections(
                ball_raw, self.ball_class_id, "ball"
            ))
        return detections


def _checkpoint_from_config(detection: dict, key: str, env_key: str) -> str:
    import os
    path = os.getenv(env_key) or detection.get(key)
    if not path:
        raise ValueError(f"Missing {key} (or env {env_key}) for local_rfdetr")
    return path


def build_detector(config: dict):
    """Build detector from config.detection backend."""
    detection = config.get("detection", {})
    backend = detection.get("backend", "local_rfdetr")
    threshold = detection.get("confidence_threshold", 0.5)
    player_class_id = detection.get("player_class_id", 0)
    ball_class_id = detection.get("ball_class_id", 1)

    if backend == "local_rfdetr":
        return LocalRFDETRDetector(
            player_checkpoint=_checkpoint_from_config(
                detection, "player_checkpoint", "PLAYER_CHECKPOINT"
            ),
            ball_checkpoint=_checkpoint_from_config(
                detection, "ball_checkpoint", "BALL_CHECKPOINT"
            ),
            confidence_threshold=threshold,
            player_class_id=player_class_id,
            ball_class_id=ball_class_id,
            enhance_ball=bool(detection.get("enhance_ball", True)),
            use_sahi=bool(detection.get("use_sahi", True)),
            use_kalman=bool(detection.get("use_kalman", False)),
        )

    if backend == "roboflow":
        import os
        from src.perception.detector import Detector

        api_key = os.getenv("ROBOFLOW_API_KEY")
        if not api_key:
            raise ValueError("ROBOFLOW_API_KEY required when detection.backend=roboflow")
        model_id = config.get("roboflow", {}).get("model_id")
        if not model_id:
            raise ValueError("roboflow.model_id required when detection.backend=roboflow")
        return Detector(model_id=model_id, api_key=api_key, confidence_threshold=threshold)

    raise ValueError(f"Unknown detection.backend: {backend}")
