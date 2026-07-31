#!/usr/bin/env python3
"""Evaluate LocalRFDETR on gold100 COCO at confidence 0.5 and 0.8."""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

import cv2

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.perception.rfdetr_local import LocalRFDETRDetector

DEFAULT_GOLD_DIR = ROOT / "data/processed/gold_sets/match1_1_100"
IOU_THRESH = 0.5


def parse_args():
    parser = argparse.ArgumentParser(description="Eval detector on gold100 COCO")
    parser.add_argument("--gold-dir", type=Path, default=DEFAULT_GOLD_DIR)
    parser.add_argument(
        "--coco",
        type=Path,
        default=None,
        help="Gold COCO (default: <gold-dir>/gold/annotations.coco.json, "
        "fallback prelabels)",
    )
    parser.add_argument("--player-checkpoint", type=Path, default=ROOT / "models/people_after_100_epochs.pth")
    parser.add_argument("--ball-checkpoint", type=Path, default=ROOT / "models/ball_89.pth")
    parser.add_argument("--thresholds", nargs="+", type=float, default=[0.5, 0.8])
    return parser.parse_args()


def resolve_coco(gold_dir: Path, coco_arg: Path | None) -> Path:
    if coco_arg is not None:
        if not coco_arg.is_file():
            raise FileNotFoundError(f"COCO not found: {coco_arg}")
        return coco_arg
    gold = gold_dir / "gold" / "annotations.coco.json"
    if gold.is_file():
        return gold
    pre = gold_dir / "prelabels" / "annotations.coco.json"
    if pre.is_file():
        print(f"Warning: using prelabels COCO (not corrected): {pre}")
        return pre
    raise FileNotFoundError(f"No COCO found under {gold_dir}")


def iou_xywh(a: List[float], b: List[float]) -> float:
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    ax2, ay2 = ax + aw, ay + ah
    bx2, by2 = bx + bw, by + bh
    ix1, iy1 = max(ax, bx), max(ay, by)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    union = aw * ah + bw * bh - inter
    if union <= 0:
        return 0.0
    return inter / union


def match_preds(
    gt_boxes: List[List[float]],
    pred_boxes: List[Tuple[List[float], float]],
) -> Tuple[int, int, int]:
    """Greedy IoU match. Returns tp, fp, fn."""
    matched_gt = set()
    tp = 0
    for box, _score in sorted(pred_boxes, key=lambda x: -x[1]):
        best_iou, best_j = 0.0, -1
        for j, gt in enumerate(gt_boxes):
            if j in matched_gt:
                continue
            score = iou_xywh(box, gt)
            if score > best_iou:
                best_iou, best_j = score, j
        if best_iou >= IOU_THRESH and best_j >= 0:
            tp += 1
            matched_gt.add(best_j)
        # else FP counted below
    fp = len(pred_boxes) - tp
    fn = len(gt_boxes) - len(matched_gt)
    return tp, fp, fn


def load_gt_by_image(coco: dict) -> Dict[int, Dict[str, List]]:
    cats = {c["id"]: c["name"] for c in coco["categories"]}
    by_image = defaultdict(lambda: {"player": [], "ball": []})
    for ann in coco["annotations"]:
        name = cats.get(ann["category_id"], "")
        if name not in ("player", "ball"):
            continue
        by_image[ann["image_id"]][name].append(ann["bbox"])
    return by_image


def pr(tp: int, fp: int, fn: int) -> Tuple[float, float]:
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    return precision, recall


def main():
    args = parse_args()
    gold_dir = args.gold_dir
    coco_path = resolve_coco(gold_dir, args.coco)
    coco = json.loads(coco_path.read_text())
    gt_by_image = load_gt_by_image(coco)
    images = {im["id"]: im for im in coco["images"]}

    # Run detector once at lowest threshold, filter in metrics
    min_thr = min(args.thresholds)
    detector = LocalRFDETRDetector(
        player_checkpoint=str(args.player_checkpoint),
        ball_checkpoint=str(args.ball_checkpoint),
        confidence_threshold=min_thr,
        player_class_id=0,
        ball_class_id=1,
    )

    preds_by_image = {}
    for image_id, im in images.items():
        path = gold_dir / "images" / im["file_name"]
        frame = cv2.imread(str(path))
        if frame is None:
            raise RuntimeError(f"Could not read {path}")
        dets = detector.detect(frame)
        preds_by_image[image_id] = dets
        print(f"image {image_id}: {len(dets)} dets")

    print(f"\nCOCO: {coco_path}")
    print(f"IoU threshold: {IOU_THRESH}")
    for thr in args.thresholds:
        totals = {
            "player": {"tp": 0, "fp": 0, "fn": 0},
            "ball": {"tp": 0, "fp": 0, "fn": 0},
        }
        for image_id, gt in gt_by_image.items():
            dets = preds_by_image.get(image_id, [])
            for cls_name in ("player", "ball"):
                pred_boxes = [
                    (list(d.bbox), d.confidence)
                    for d in dets
                    if d.class_name == cls_name and d.confidence >= thr
                ]
                tp, fp, fn = match_preds(gt[cls_name], pred_boxes)
                totals[cls_name]["tp"] += tp
                totals[cls_name]["fp"] += fp
                totals[cls_name]["fn"] += fn
        print(f"\n=== conf >= {thr} ===")
        for cls_name in ("player", "ball"):
            t = totals[cls_name]
            p, r = pr(t["tp"], t["fp"], t["fn"])
            print(
                f"{cls_name}: P={p:.3f} R={r:.3f} "
                f"(tp={t['tp']} fp={t['fp']} fn={t['fn']})"
            )


if __name__ == "__main__":
    main()
