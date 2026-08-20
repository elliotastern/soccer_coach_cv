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


def _inter_xywh(a, b) -> float:
    ax, ay, aw, ah = [float(v) for v in a]
    bx, by, bw, bh = [float(v) for v in b]
    ix1, iy1 = max(ax, bx), max(ay, by)
    ix2, iy2 = min(ax + aw, bx + bw), min(ay + ah, by + bh)
    return max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)


def _iou_xywh(a, b) -> float:
    ax, ay, aw, ah = [float(v) for v in a]
    bx, by, bw, bh = [float(v) for v in b]
    inter = _inter_xywh(a, b)
    union = aw * ah + bw * bh - inter
    return inter / union if union > 0 else 0.0


def _overlap_same_object(a, b) -> float:
    """IoU or containment (inter/min area). Nested torso/full-body boxes often have low IoU."""
    ax, ay, aw, ah = [float(v) for v in a]
    bx, by, bw, bh = [float(v) for v in b]
    inter = _inter_xywh(a, b)
    area_a, area_b = aw * ah, bw * bh
    if area_a <= 0 or area_b <= 0:
        return 0.0
    iou = inter / (area_a + area_b - inter)
    iomin = inter / min(area_a, area_b)
    return max(iou, iomin)


def _x_overlap_frac(a, b) -> float:
    ax, _, aw, _ = [float(v) for v in a]
    bx, _, bw, _ = [float(v) for v in b]
    inter = max(0.0, min(ax + aw, bx + bw) - max(ax, bx))
    denom = min(aw, bw)
    return inter / denom if denom > 0 else 0.0


def _y_overlap_frac(a, b) -> float:
    _, ay, _, ah = [float(v) for v in a]
    _, by, _, bh = [float(v) for v in b]
    inter = max(0.0, min(ay + ah, by + bh) - max(ay, by))
    denom = min(ah, bh)
    return inter / denom if denom > 0 else 0.0


def _is_duplicate_player(a, b, overlap_thr: float) -> bool:
    """True when two boxes are the same person (nest, overlap, or torso/legs split)."""
    if _overlap_same_object(a, b) >= overlap_thr:
        return True
    ax, ay, aw, ah = [float(v) for v in a]
    bx, by, bw, bh = [float(v) for v in b]
    x_ov = _x_overlap_frac(a, b)
    y_ov = _y_overlap_frac(a, b)
    if x_ov < 0.30:
        return False
    cax, cay = ax + aw / 2.0, ay + ah / 2.0
    cbx, cby = bx + bw / 2.0, by + bh / 2.0
    dist = ((cax - cbx) ** 2 + (cay - cby) ** 2) ** 0.5
    ref_h = max(ah, bh)
    # Center close relative to body height (upper+lower splits)
    if dist < 1.15 * ref_h and x_ov >= 0.35:
        return True
    # Stacked / overlapping in y with shared column (torso vs legs)
    if x_ov >= 0.40 and y_ov >= 0.20:
        return True
    # Nearly touching vertically in same column
    gap = max(0.0, max(ay, by) - min(ay + ah, by + bh))
    if x_ov >= 0.45 and gap < 0.25 * (0.5 * (ah + bh)):
        return True
    return False


def nms_class(dets: List[Detection], overlap_thr: float = 0.30) -> List[Detection]:
    """Keep one box per person; prefer higher conf, then taller (full body)."""
    ordered = sorted(
        dets,
        key=lambda d: (float(d.confidence), float(d.bbox[3])),
        reverse=True,
    )
    kept: List[Detection] = []
    for det in ordered:
        rival_i = None
        for i, k in enumerate(kept):
            if _is_duplicate_player(det.bbox, k.bbox, overlap_thr):
                rival_i = i
                break
        if rival_i is None:
            kept.append(det)
            continue
        rival = kept[rival_i]
        # Swap in taller box when conf is close (coach: full body > torso stub)
        taller = float(det.bbox[3]) > float(rival.bbox[3]) * 1.08
        close = float(det.confidence) >= float(rival.confidence) * 0.70
        if taller and close:
            kept[rival_i] = det
    return kept


def nms_by_class(
    dets: List[Detection],
    player_iou: float = 0.30,
    ball_iou: float = 0.4,
) -> List[Detection]:
    players = [d for d in dets if d.class_name == "player" or int(d.class_id) == 0]
    balls = [d for d in dets if d.class_name == "ball" or int(d.class_id) == 1]
    other = [
        d
        for d in dets
        if d not in players and d not in balls
    ]
    return nms_class(players, player_iou) + nms_class(balls, ball_iou) + other


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
        player_nms_iou: float = 0.30,
        ball_nms_iou: float = 0.4,
    ):
        self.confidence_threshold = confidence_threshold
        self.player_class_id = player_class_id
        self.ball_class_id = ball_class_id
        self.enhance_ball = enhance_ball
        self.use_sahi = use_sahi
        self.use_kalman = use_kalman
        self.player_nms_iou = float(player_nms_iou)
        self.ball_nms_iou = float(ball_nms_iou)
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
        return nms_by_class(
            detections,
            player_iou=self.player_nms_iou,
            ball_iou=self.ball_nms_iou,
        )

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
            player_nms_iou=float(detection.get("player_nms_iou", 0.30)),
            ball_nms_iou=float(detection.get("ball_nms_iou", 0.4)),
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
