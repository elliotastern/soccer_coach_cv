"""Advanced ball prelabel helpers: SAHI tiling, size filters, Kalman coasting."""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

import numpy as np
from PIL import Image

from src.state.types import Detection


def xywh_to_xyxy(bbox: Tuple[float, float, float, float]) -> Tuple[float, float, float, float]:
    x, y, w, h = bbox
    return x, y, x + w, y + h


def iou_xywh(a: Tuple[float, float, float, float], b: Tuple[float, float, float, float]) -> float:
    ax1, ay1, ax2, ay2 = xywh_to_xyxy(a)
    bx1, by1, bx2, by2 = xywh_to_xyxy(b)
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    union = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1) + max(0.0, bx2 - bx1) * max(0.0, by2 - by1) - inter
    return inter / union if union > 0 else 0.0


def nms_balls(dets: Sequence[Detection], iou_thr: float = 0.4) -> List[Detection]:
    ordered = sorted(dets, key=lambda d: d.confidence, reverse=True)
    kept: List[Detection] = []
    for det in ordered:
        if all(iou_xywh(det.bbox, k.bbox) < iou_thr for k in kept):
            kept.append(det)
    return kept


def filter_ball_geometry(
    dets: Sequence[Detection],
    min_side: float = 4.0,
    max_side: float = 120.0,
    min_aspect: float = 0.35,
    max_aspect: float = 2.8,
    image_width: Optional[float] = None,
) -> List[Detection]:
    # Scale limits with resolution (defaults tuned for ~1920px review frames)
    scale = (image_width / 1920.0) if image_width else 1.0
    min_side = min_side * scale
    max_side = max_side * scale
    out = []
    for det in dets:
        _, _, w, h = det.bbox
        if w < min_side or h < min_side or w > max_side or h > max_side:
            continue
        aspect = w / h if h > 0 else 999.0
        if aspect < min_aspect or aspect > max_aspect:
            continue
        out.append(det)
    return out


def topk_balls(dets: Sequence[Detection], k: int = 1) -> List[Detection]:
    if k <= 0:
        return list(dets)
    return sorted(dets, key=lambda d: d.confidence, reverse=True)[:k]


def _tile_origins(length: int, slice_size: int, overlap: float) -> List[int]:
    if length <= slice_size:
        return [0]
    step = max(1, int(slice_size * (1.0 - overlap)))
    origins = list(range(0, max(1, length - slice_size + 1), step))
    last = length - slice_size
    if origins[-1] != last:
        origins.append(last)
    return origins


def slice_grid(
    width: int,
    height: int,
    slice_size: int = 640,
    overlap: float = 0.2,
) -> List[Tuple[int, int, int, int]]:
    boxes = []
    for y0 in _tile_origins(height, slice_size, overlap):
        for x0 in _tile_origins(width, slice_size, overlap):
            x1 = min(width, x0 + slice_size)
            y1 = min(height, y0 + slice_size)
            boxes.append((x0, y0, x1, y1))
    return boxes


def predict_balls_fullframe(model, pil_image: Image.Image, threshold: float, class_id: int = 1) -> List[Detection]:
    raw = model.predict(pil_image, threshold=threshold)
    if not hasattr(raw, "class_id"):
        return []
    out = []
    for i in range(len(raw.class_id)):
        x1, y1, x2, y2 = map(float, raw.xyxy[i])
        w, h = x2 - x1, y2 - y1
        if w <= 0 or h <= 0:
            continue
        out.append(Detection(
            class_id=class_id,
            confidence=float(raw.confidence[i]),
            bbox=(x1, y1, w, h),
            class_name="ball",
        ))
    return out


def predict_balls_sahi(
    model,
    pil_image: Image.Image,
    threshold: float,
    slice_size: int = 640,
    overlap: float = 0.2,
    class_id: int = 1,
    include_fullframe: bool = True,
) -> List[Detection]:
    width, height = pil_image.size
    merged: List[Detection] = []
    if include_fullframe:
        merged.extend(predict_balls_fullframe(model, pil_image, threshold, class_id))
    if width > slice_size or height > slice_size:
        for x0, y0, x1, y1 in slice_grid(width, height, slice_size, overlap):
            crop = pil_image.crop((x0, y0, x1, y1))
            for det in predict_balls_fullframe(model, crop, threshold, class_id):
                x, y, w, h = det.bbox
                merged.append(Detection(
                    class_id=det.class_id,
                    confidence=det.confidence,
                    bbox=(x + x0, y + y0, w, h),
                    class_name="ball",
                ))
    return nms_balls(merged, iou_thr=0.4)


def sahi_recover_only(
    fullframe: Sequence[Detection],
    tile_dets: Sequence[Detection],
    max_iou: float = 0.1,
) -> List[Detection]:
    """Keep tile dets only if they do not overlap existing fullframe boxes (FN recovery)."""
    recovered = []
    for det in tile_dets:
        if all(iou_xywh(det.bbox, f.bbox) < max_iou for f in fullframe):
            recovered.append(det)
    return recovered


@dataclass
class KalmanBallState:
    x: float
    y: float
    vx: float = 0.0
    vy: float = 0.0
    conf: float = 0.0
    age: int = 0
    hits: int = 0
    time_since_update: int = 0


class KalmanBallTracker:
    """Constant-velocity Kalman for a single ball hypothesis (prelabel assist)."""

    def __init__(
        self,
        process_var: float = 25.0,
        meas_var: float = 16.0,
        max_coast: int = 8,
        gate_px: float = 120.0,
    ):
        self.process_var = process_var
        self.meas_var = meas_var
        self.max_coast = max_coast
        self.gate_px = gate_px
        self.state: Optional[KalmanBallState] = None
        self.p = np.eye(4) * 50.0

    def _center(self, det: Detection) -> Tuple[float, float]:
        x, y, w, h = det.bbox
        return x + w * 0.5, y + h * 0.5

    def predict(self) -> Optional[Tuple[float, float]]:
        if self.state is None:
            return None
        self.state.x += self.state.vx
        self.state.y += self.state.vy
        self.state.age += 1
        self.state.time_since_update += 1
        q = self.process_var
        self.p = self.p + np.diag([q, q, q * 0.5, q * 0.5])
        return self.state.x, self.state.y

    def update(self, det: Optional[Detection]) -> Optional[Detection]:
        if det is None:
            if self.state is None:
                return None
            # Require ≥2 hits before coasting; drop after max_coast
            if self.state.hits < 2 or self.state.time_since_update > self.max_coast:
                self.state = None
                return None
            side = 12.0
            conf = max(0.15, self.state.conf * 0.85)
            return Detection(
                class_id=1,
                confidence=conf,
                bbox=(self.state.x - side / 2, self.state.y - side / 2, side, side),
                class_name="ball",
            )
        cx, cy = self._center(det)
        if self.state is None:
            self.state = KalmanBallState(x=cx, y=cy, conf=det.confidence, hits=1)
            self.p = np.eye(4) * 20.0
            return det
        # Gate distant measurements as new init
        dist = ((cx - self.state.x) ** 2 + (cy - self.state.y) ** 2) ** 0.5
        if dist > self.gate_px and self.state.time_since_update == 0:
            # Prefer higher-confidence measurement near track; else re-init
            if det.confidence < self.state.conf:
                return Detection(
                    class_id=1,
                    confidence=self.state.conf,
                    bbox=(self.state.x - 6, self.state.y - 6, 12, 12),
                    class_name="ball",
                )
            self.state = KalmanBallState(x=cx, y=cy, conf=det.confidence, hits=1)
            return det
        # Simple gain blend (scalar Kalman-ish)
        k = self.meas_var / (self.meas_var + max(self.p[0, 0], 1.0))
        self.state.vx = (1 - k) * self.state.vx + k * (cx - self.state.x)
        self.state.vy = (1 - k) * self.state.vy + k * (cy - self.state.y)
        self.state.x = (1 - k) * self.state.x + k * cx
        self.state.y = (1 - k) * self.state.y + k * cy
        self.state.conf = max(self.state.conf * 0.9, det.confidence)
        self.state.hits += 1
        self.state.time_since_update = 0
        self.p *= (1 - k)
        x, y, w, h = det.bbox
        return Detection(
            class_id=det.class_id,
            confidence=det.confidence,
            bbox=(self.state.x - w / 2, self.state.y - h / 2, w, h),
            class_name="ball",
        )

    def step(self, dets: Sequence[Detection]) -> List[Detection]:
        self.predict()
        if not dets:
            coasted = self.update(None)
            return [coasted] if coasted is not None else []
        best = max(dets, key=lambda d: d.confidence)
        if self.state is not None:
            # Prefer measurement closest to prediction within gate
            cx0, cy0 = self.state.x, self.state.y
            gated = []
            for d in dets:
                cx, cy = self._center(d)
                dist = ((cx - cx0) ** 2 + (cy - cy0) ** 2) ** 0.5
                if dist <= self.gate_px:
                    gated.append((dist, d))
            if gated:
                best = min(gated, key=lambda t: (t[0], -t[1].confidence))[1]
        out = self.update(best)
        return [out] if out is not None else []


@dataclass
class BallPrelabelConfig:
    threshold: float = 0.30
    tile_threshold: Optional[float] = None  # defaults to max(threshold, 0.4)
    use_sahi: bool = False
    sahi_fallback_only: bool = True  # tiles only if fullframe empty
    sahi_recover_only: bool = True  # never replace fullframe; only add non-overlapping tiles
    slice_size: int = 960
    overlap: float = 0.2
    use_size_filter: bool = True
    use_multiscale: bool = False
    multiscale_factor: float = 1.5
    topk: int = 2
    use_kalman: bool = False
    max_side: float = 120.0
    min_side: float = 4.0


class BallPrelabeler:
    """Ball-only enhanced detector for prelabeling."""

    def __init__(self, ball_model, config: Optional[BallPrelabelConfig] = None, class_id: int = 1):
        self.model = ball_model
        self.config = config or BallPrelabelConfig()
        self.class_id = class_id
        self.kalman = KalmanBallTracker() if self.config.use_kalman else None

    def detect_bgr(self, frame_bgr: np.ndarray) -> List[Detection]:
        import cv2

        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        return self.detect_pil(Image.fromarray(rgb))

    def detect_pil(self, pil_image: Image.Image) -> List[Detection]:
        cfg = self.config
        tile_thr = cfg.tile_threshold if cfg.tile_threshold is not None else max(cfg.threshold, 0.4)
        img_w = float(pil_image.size[0])
        dets = predict_balls_fullframe(
            self.model, pil_image, threshold=cfg.threshold, class_id=self.class_id
        )
        if cfg.use_multiscale:
            factor = cfg.multiscale_factor
            up_w = int(pil_image.size[0] * factor)
            up_h = int(pil_image.size[1] * factor)
            up = pil_image.resize((up_w, up_h), Image.BILINEAR)
            for det in predict_balls_fullframe(
                self.model, up, threshold=cfg.threshold, class_id=self.class_id
            ):
                x, y, w, h = det.bbox
                dets.append(Detection(
                    class_id=det.class_id,
                    confidence=det.confidence,
                    bbox=(x / factor, y / factor, w / factor, h / factor),
                    class_name="ball",
                ))
            dets = nms_balls(dets, iou_thr=0.4)
        if cfg.use_size_filter:
            dets = filter_ball_geometry(
                dets, min_side=cfg.min_side, max_side=cfg.max_side, image_width=img_w
            )
        need_sahi = cfg.use_sahi and (not cfg.sahi_fallback_only or len(dets) == 0)
        if need_sahi:
            full_before = list(dets)
            tile_dets = predict_balls_sahi(
                self.model,
                pil_image,
                threshold=tile_thr,
                slice_size=cfg.slice_size,
                overlap=cfg.overlap,
                class_id=self.class_id,
                include_fullframe=False,
            )
            if cfg.use_size_filter:
                tile_dets = filter_ball_geometry(
                    tile_dets, min_side=cfg.min_side, max_side=cfg.max_side, image_width=img_w
                )
            if cfg.sahi_recover_only:
                tile_dets = sahi_recover_only(full_before, tile_dets, max_iou=0.1)
            dets = nms_balls(list(full_before) + list(tile_dets), iou_thr=0.4)
        dets = topk_balls(dets, k=cfg.topk)
        if self.kalman is not None:
            dets = self.kalman.step(dets)
        return dets

    def reset(self) -> None:
        if self.config.use_kalman:
            self.kalman = KalmanBallTracker()