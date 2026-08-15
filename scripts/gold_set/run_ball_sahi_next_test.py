#!/usr/bin/env python3
"""P10 Top Left: next-gen SAHI dashboard (Gemini D1–D10), vs fallback-only.

Same clip as ball_postprocessing_test (0:26–0:31, P10). Inference-only.
Encoder-logit / entropy ideas use low-threshold detect proxies (RF-DETR
query hooks not exposed). Player-context uses frame-diff motion ROIs.
Writes reports/eval_match2_v10/ball_sahi_next_test/. Never trains.
"""
from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts" / "gold_set"))

from run_5x5_ball_clips import encode_browser_mp4, paint_view  # noqa: E402
from run_ball_postprocessing_test import (  # noqa: E402
    CKPT,
    END_CLOCK,
    OVERLAY_WIDTH,
    START_CLOCK,
    ensure_source,
    parse_clock,
    write_html,
    write_summary,
)
from eval_match2_v10_video_system import det_tuple  # noqa: E402
from src.perception.ball_prelabel import (  # noqa: E402
    Detection,
    filter_ball_geometry,
    iou_xywh,
    nms_balls,
    predict_balls_fullframe,
    slice_grid,
    topk_balls,
)
from src.perception.rfdetr_local import load_ball_model  # noqa: E402

OUT_DEFAULT = ROOT / "reports/eval_match2_v10/ball_sahi_next_test"
THR = 0.30
TILE = 960
OVERLAP = 0.2
MIN_SIDE, MAX_SIDE = 4.0, 240.0


def variant_specs() -> list[dict]:
    return [
        {
            "id": "D1_sparse_logit",
            "title": "1. D1 Sparse logit tiles",
            "why": "On empty full-frame, tile only around subthreshold (0.10–0.29) detects as logit proxy.",
            "kind": "sparse_logit",
        },
        {
            "id": "D2_wbf_merge",
            "title": "2. D2 Fallback + WBF merge",
            "why": "Standard fallback grid; Weighted Boxes Fusion instead of hard NMS.",
            "kind": "wbf",
        },
        {
            "id": "D3_diou_merge",
            "title": "3. D3 Fallback + Cluster-DIoU-NMS",
            "why": "Centroid-aware suppress for tiny-ball tile duplicates (<15px).",
            "kind": "diou",
        },
        {
            "id": "D4_temporal_ema",
            "title": "4. D4 Temporal EMA motion tile",
            "why": "One tile on EMA-projected (x,y) when empty — verify with network, no Kalman coast.",
            "kind": "ema",
        },
        {
            "id": "D5_soft_edge",
            "title": "5. D5 Soft edge penalty",
            "why": "Down-weight tile dets near tile borders (truncation FPs), then NMS.",
            "kind": "soft_edge",
        },
        {
            "id": "D6_player_context",
            "title": "6. D6 Motion-ROI tiles (player proxy)",
            "why": "No person head on ball ckpt — tile only where frame-diff motion is high.",
            "kind": "motion_roi",
        },
        {
            "id": "D7_adaptive_asahi",
            "title": "7. D7 Adaptive ASAHI grid",
            "why": "Far pitch (top): 640px tiles; near (bottom): 1024px — then WBF.",
            "kind": "asahi",
        },
        {
            "id": "D8_sr_conditional",
            "title": "8. D8 Conditional SR tile",
            "why": "One tile on best subthreshold center, 1.5× upscale, detect, scale back.",
            "kind": "sr",
        },
        {
            "id": "D9_dotd_tracker",
            "title": "9. D9 DotD track gate",
            "why": "Fallback grid + WBF, then emit only if linked by Dot Distance to prior.",
            "kind": "dotd",
        },
        {
            "id": "D10_entropy_crop",
            "title": "10. D10 Entropy/uncertainty crop",
            "why": "Tile on detection closest to conf=0.5 (uncertainty proxy) when empty.",
            "kind": "entropy",
        },
    ]


def to_pil(frame_bgr):
    return Image.fromarray(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB))


def geom(dets, w):
    return filter_ball_geometry(dets, min_side=MIN_SIDE, max_side=MAX_SIDE, image_width=float(w))


def full_at(model, pil, thr=THR):
    return predict_balls_fullframe(model, pil, threshold=thr, class_id=1)


def center_xy(det: Detection):
    x, y, w, h = det.bbox
    return x + w * 0.5, y + h * 0.5


def crop_tile(pil: Image.Image, cx: float, cy: float, size: int = TILE):
    w, h = pil.size
    half = size // 2
    x0 = int(max(0, min(w - size, round(cx) - half)))
    y0 = int(max(0, min(h - size, round(cy) - half)))
    x1, y1 = min(w, x0 + size), min(h, y0 + size)
    return (x0, y0, x1, y1), pil.crop((x0, y0, x1, y1))


def detect_on_tiles(model, pil, boxes, thr=THR):
    out = []
    for x0, y0, x1, y1 in boxes:
        crop = pil.crop((x0, y0, x1, y1))
        for det in full_at(model, crop, thr):
            x, y, bw, bh = det.bbox
            out.append(
                Detection(
                    class_id=1,
                    confidence=det.confidence,
                    bbox=(x + x0, y + y0, bw, bh),
                    class_name="ball",
                )
            )
    return out


def standard_fallback_tiles(pil: Image.Image, slice_size=TILE, overlap=OVERLAP):
    return slice_grid(pil.size[0], pil.size[1], slice_size, overlap)


def wbf_merge(dets: list, iou_thr=0.1, dist_px=40.0) -> list:
    if not dets:
        return []
    order = sorted(dets, key=lambda d: d.confidence, reverse=True)
    used = [False] * len(order)
    fused = []
    for i, a in enumerate(order):
        if used[i]:
            continue
        cluster = [a]
        used[i] = True
        ax, ay = center_xy(a)
        for j in range(i + 1, len(order)):
            if used[j]:
                continue
            b = order[j]
            bx, by = center_xy(b)
            close = ((ax - bx) ** 2 + (ay - by) ** 2) ** 0.5 <= dist_px
            if iou_xywh(a.bbox, b.bbox) >= iou_thr or close:
                cluster.append(b)
                used[j] = True
        sw = sum(d.confidence for d in cluster)
        if sw <= 0:
            continue
        x = sum(center_xy(d)[0] * d.confidence for d in cluster) / sw
        y = sum(center_xy(d)[1] * d.confidence for d in cluster) / sw
        w = sum(d.bbox[2] * d.confidence for d in cluster) / sw
        h = sum(d.bbox[3] * d.confidence for d in cluster) / sw
        conf = max(d.confidence for d in cluster)
        fused.append(
            Detection(class_id=1, confidence=conf, bbox=(x - w / 2, y - h / 2, w, h), class_name="ball")
        )
    return fused


def diou(a, b) -> float:
    ax, ay = center_xy(a)
    bx, by = center_xy(b)
    inter = iou_xywh(a.bbox, b.bbox)
    dist = (ax - bx) ** 2 + (ay - by) ** 2
    x1 = min(a.bbox[0], b.bbox[0])
    y1 = min(a.bbox[1], b.bbox[1])
    x2 = max(a.bbox[0] + a.bbox[2], b.bbox[0] + b.bbox[2])
    y2 = max(a.bbox[1] + a.bbox[3], b.bbox[1] + b.bbox[3])
    c2 = max((x2 - x1) ** 2 + (y2 - y1) ** 2, 1e-6)
    return inter - dist / c2


def cluster_diou_nms(dets: list, dist_px=15.0) -> list:
    if not dets:
        return []
    order = sorted(dets, key=lambda d: d.confidence, reverse=True)
    keep = []
    for det in order:
        cx, cy = center_xy(det)
        drop = False
        for k in keep:
            kx, ky = center_xy(k)
            if ((cx - kx) ** 2 + (cy - ky) ** 2) ** 0.5 <= dist_px or diou(det, k) > 0.0:
                drop = True
                break
        if not drop:
            keep.append(det)
    return keep


def soft_edge_penalize(dets, tile_box, margin_frac=0.10):
    x0, y0, x1, y1 = tile_box
    tw, th = max(x1 - x0, 1), max(y1 - y0, 1)
    out = []
    for det in dets:
        cx, cy = center_xy(det)
        lx, ly = cx - x0, cy - y0
        d_edge = min(lx / tw, ly / th, 1 - lx / tw, 1 - ly / th)
        conf = det.confidence
        if d_edge < margin_frac:
            conf *= max(0.05, d_edge / margin_frac)
        out.append(
            Detection(class_id=1, confidence=conf, bbox=det.bbox, class_name="ball")
        )
    return out


def motion_rois(prev_gray, gray, cell=480, top_k=4):
    if prev_gray is None:
        return []
    diff = cv2.absdiff(prev_gray, gray)
    h, w = diff.shape
    scores = []
    for y0 in range(0, h, cell):
        for x0 in range(0, w, cell):
            patch = diff[y0 : min(h, y0 + cell), x0 : min(w, x0 + cell)]
            scores.append((float(patch.mean()), x0 + cell // 2, y0 + cell // 2))
    scores.sort(reverse=True)
    return [(x, y) for _s, x, y in scores[:top_k]]


def asahi_grid(w, h):
    boxes = []
    mid = h // 2
    for x0, y0, x1, y1 in slice_grid(w, mid, 640, 0.25):
        boxes.append((x0, y0, x1, y1))
    for x0, y0, x1, y1 in slice_grid(w, h - mid, 1024, 0.20):
        boxes.append((x0, y0 + mid, x1, y1 + mid))
    return boxes


def detect_variant(model, frame, kind: str, state: dict) -> list:
    pil = to_pil(frame)
    w, h = pil.size
    full = geom(full_at(model, pil, THR), w)
    low = geom(full_at(model, pil, 0.05), w)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    if full:
        state["ema"] = _update_ema(state.get("ema"), full[0])
        state["last"] = full[0]
        state["prev_gray"] = gray
        return topk_balls(full, 2)

    if kind == "sparse_logit":
        seeds = [d for d in low if 0.10 <= d.confidence <= 0.29]
        seeds = sorted(seeds, key=lambda d: d.confidence, reverse=True)[:2]
        boxes = []
        for d in seeds:
            box, _ = crop_tile(pil, *center_xy(d))
            boxes.append(box)
        if not boxes:
            boxes = standard_fallback_tiles(pil)[:4]
        tile = detect_on_tiles(model, pil, boxes)
        return topk_balls(geom(nms_balls(tile, 0.4), w), 2)

    if kind == "wbf":
        tile = detect_on_tiles(model, pil, standard_fallback_tiles(pil))
        return topk_balls(geom(wbf_merge(tile), w), 2)

    if kind == "diou":
        tile = detect_on_tiles(model, pil, standard_fallback_tiles(pil))
        return topk_balls(geom(cluster_diou_nms(tile), w), 2)

    if kind == "ema":
        ema = state.get("ema")
        if ema is None:
            boxes = standard_fallback_tiles(pil)[:6]
        else:
            cx = ema["x"] + ema["vx"]
            cy = ema["y"] + ema["vy"]
            box, _ = crop_tile(pil, cx, cy)
            boxes = [box]
        tile = detect_on_tiles(model, pil, boxes)
        dets = topk_balls(geom(nms_balls(tile, 0.4), w), 2)
        if dets:
            state["ema"] = _update_ema(ema, dets[0])
            state["last"] = dets[0]
        state["prev_gray"] = gray
        return dets

    if kind == "soft_edge":
        out = []
        for box in standard_fallback_tiles(pil):
            raw = detect_on_tiles(model, pil, [box])
            out.extend(soft_edge_penalize(raw, box))
        return topk_balls(geom(nms_balls(out, 0.4), w), 2)

    if kind == "motion_roi":
        rois = motion_rois(state.get("prev_gray"), gray)
        state["prev_gray"] = gray
        boxes = []
        for cx, cy in rois:
            box, _ = crop_tile(pil, cx, cy)
            boxes.append(box)
        if not boxes:
            boxes = standard_fallback_tiles(pil)[:4]
        tile = detect_on_tiles(model, pil, boxes)
        return topk_balls(geom(nms_balls(tile, 0.4), w), 2)

    if kind == "asahi":
        tile = detect_on_tiles(model, pil, asahi_grid(w, h))
        return topk_balls(geom(wbf_merge(tile), w), 2)

    if kind == "sr":
        seeds = sorted(low, key=lambda d: d.confidence, reverse=True)
        if seeds:
            cx, cy = center_xy(seeds[0])
        elif state.get("ema"):
            cx, cy = state["ema"]["x"], state["ema"]["y"]
        else:
            cx, cy = w / 2, h / 2
        (x0, y0, x1, y1), crop = crop_tile(pil, cx, cy)
        up = crop.resize((int(crop.size[0] * 1.5), int(crop.size[1] * 1.5)), Image.BILINEAR)
        raw = full_at(model, up, THR)
        scaled = []
        for det in raw:
            x, y, bw, bh = det.bbox
            scaled.append(
                Detection(
                    class_id=1,
                    confidence=det.confidence,
                    bbox=(x / 1.5 + x0, y / 1.5 + y0, bw / 1.5, bh / 1.5),
                    class_name="ball",
                )
            )
        dets = topk_balls(geom(nms_balls(scaled, 0.4), w), 2)
        if dets:
            state["ema"] = _update_ema(state.get("ema"), dets[0])
            state["last"] = dets[0]
        state["prev_gray"] = gray
        return dets

    if kind == "dotd":
        tile = detect_on_tiles(model, pil, standard_fallback_tiles(pil))
        merged = topk_balls(geom(wbf_merge(tile), w), 2)
        prev = state.get("last")
        if prev is None:
            state["last"] = merged[0] if merged else None
            state["prev_gray"] = gray
            return merged
        linked = []
        px, py = center_xy(prev)
        scale = 120.0
        for det in merged:
            cx, cy = center_xy(det)
            dist = math.hypot(cx - px, cy - py)
            score = max(0.0, 1.0 - dist / scale)
            if score >= 0.35:
                linked.append(det)
        if linked:
            state["last"] = linked[0]
            state["ema"] = _update_ema(state.get("ema"), linked[0])
        state["prev_gray"] = gray
        return linked

    if kind == "entropy":
        # uncertainty proxy: conf nearest 0.5
        if low:
            seed = min(low, key=lambda d: abs(d.confidence - 0.5))
            box, _ = crop_tile(pil, *center_xy(seed))
            boxes = [box]
        else:
            boxes = standard_fallback_tiles(pil)[:4]
        tile = detect_on_tiles(model, pil, boxes)
        dets = topk_balls(geom(nms_balls(tile, 0.4), w), 2)
        if dets:
            state["last"] = dets[0]
            state["ema"] = _update_ema(state.get("ema"), dets[0])
        state["prev_gray"] = gray
        return dets

    raise ValueError(kind)


def _update_ema(ema, det: Detection):
    cx, cy = center_xy(det)
    if ema is None:
        return {"x": cx, "y": cy, "vx": 0.0, "vy": 0.0}
    alpha = 0.4
    vx = cx - ema["x"]
    vy = cy - ema["y"]
    return {
        "x": alpha * cx + (1 - alpha) * ema["x"],
        "y": alpha * cy + (1 - alpha) * ema["y"],
        "vx": alpha * vx + (1 - alpha) * ema["vx"],
        "vy": alpha * vy + (1 - alpha) * ema["vy"],
    }


def top_pred(dets):
    if not dets:
        return None
    rows = [det_tuple(d) for d in dets]
    rows.sort(key=lambda x: -x[1])
    return rows[0]


def run_variant(model, src: Path, ov_path: Path, spec: dict, stride: int, width: int) -> dict:
    cap = cv2.VideoCapture(str(src))
    if not cap.isOpened():
        raise RuntimeError(f"open failed: {src}")
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 30.0)
    writer = None
    state = {}
    n_frames = n_raw = n_emit = 0
    confs = []
    last_raw = last_emit = None
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        n_frames += 1
        if (n_frames - 1) % max(1, stride) == 0:
            dets = detect_variant(model, frame, spec["kind"], state)
            raw = top_pred(dets)
            emit = raw if raw is not None and raw[1] >= 0.80 else None
            last_raw, last_emit = raw, emit
        if last_raw is not None:
            n_raw += 1
        if last_emit is not None:
            n_emit += 1
            confs.append(float(last_emit[1]))
        vis = paint_view(frame, last_raw, last_emit, "P10", width)
        if writer is None:
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            writer = cv2.VideoWriter(str(ov_path), fourcc, fps, (vis.shape[1], vis.shape[0]))
        writer.write(vis)
        top = last_emit[1] if last_emit is not None else (last_raw[1] if last_raw else 0.0)
        print(
            f"  {spec['id']} f={n_frames:03d} raw={last_raw is not None} "
            f"emit={last_emit is not None} top={top:.3f}",
            flush=True,
        )
    cap.release()
    if writer is not None:
        writer.release()
        encode_browser_mp4(ov_path)
    return {
        "id": spec["id"],
        "title": spec["title"],
        "why": spec["why"],
        "mode": spec["kind"],
        "n_frames": n_frames,
        "n_raw_hits": n_raw,
        "n_emit_hold": n_emit,
        "emit_rate": (n_emit / n_frames) if n_frames else 0.0,
        "raw_rate": (n_raw / n_frames) if n_frames else 0.0,
        "mean_emit_conf": (sum(confs) / len(confs)) if confs else None,
        "overlay": f"overlay/{ov_path.name}",
    }


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--ball-checkpoint", type=Path, default=CKPT)
    p.add_argument("--out", type=Path, default=OUT_DEFAULT)
    p.add_argument("--stride", type=int, default=2)
    p.add_argument("--overlay-width", type=int, default=OVERLAY_WIDTH)
    p.add_argument("--skip-extract", action="store_true")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    out = args.out
    out.mkdir(parents=True, exist_ok=True)
    (out / "overlay").mkdir(parents=True, exist_ok=True)
    src = ensure_source(out, args.skip_extract)
    print(f"source {src}", flush=True)
    model = load_ball_model(str(args.ball_checkpoint))
    specs = variant_specs()
    if len(specs) != 10:
        raise RuntimeError(f"expected 10, got {len(specs)}")
    variants = []
    for spec in specs:
        ov = out / "overlay" / f"{spec['id']}.mp4"
        print(f"variant {spec['id']} → {ov}", flush=True)
        variants.append(run_variant(model, src, ov, spec, args.stride, args.overlay_width))
    payload = {
        "title": "ball_sahi_next_test",
        "clock": f"{START_CLOCK}–{END_CLOCK}",
        "start_sec": parse_clock(START_CLOCK),
        "end_sec": parse_clock(END_CLOCK),
        "camera": "P10",
        "source": str(src),
        "checkpoint": str(args.ball_checkpoint),
        "stride": args.stride,
        "ranking_note": "Gemini next-gen SAHI D1–D10 vs fallback-only winner",
        "page_note": (
            "Next SAHI attempts from deep research (sparse logit, WBF, DIoU, EMA tile, "
            "soft-edge, motion ROI, ASAHI grid, conditional SR, DotD gate, entropy crop). "
            "D1/D10 use low-threshold detects as encoder-query proxies; D6 uses motion ROIs "
            "(no person head on ball checkpoint). Same P10 Top Left clip."
        ),
        "variants": variants,
    }
    summary = write_summary(out, payload)
    write_html(out, payload)
    print(f"wrote {summary}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
