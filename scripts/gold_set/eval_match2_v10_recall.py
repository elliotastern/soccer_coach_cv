#!/usr/bin/env python3
"""Match 2 gold recall/precision with no 0.80 emit gate.

Scores held-out gold 50 at detect 0.3 and 0.5. Gold XML/COCO is source of truth.
Does not train. Does not use train100.
"""
from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

import cv2

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts" / "gold_set"))

from eval_poc_ball_metrics import detect_balls, match_preds, pr
from src.perception.rfdetr_local import load_ball_model

GOLD = ROOT / "data/processed/gold_sets/match2_gold_frames"
CKPT = ROOT / "models/v10_snaps/post_train/checkpoint.pth"
OUT = ROOT / "reports/eval_match2_v10/recall_no_emit_gate.json"
IOU_THR = 0.5


def load_items(gold_dir: Path):
    coco = json.loads((gold_dir / "gold" / "annotations.coco.json").read_text())
    cats = {c["id"]: c["name"] for c in coco["categories"]}
    by_image = defaultdict(list)
    for a in coco["annotations"]:
        if cats.get(a["category_id"]) == "ball":
            by_image[a["image_id"]].append(a["bbox"])
    id_by_name = {im["file_name"]: im for im in coco["images"]}
    man = json.loads((gold_dir / "manifest.json").read_text())
    items = []
    for row in man["frames"]:
        im = id_by_name[row["image"]]
        path = gold_dir / "images" / row["image"]
        if not path.is_file():
            raise FileNotFoundError(path)
        items.append({
            "camera": row["camera"],
            "image": row["image"],
            "path": path,
            "gt": by_image[im["id"]],
        })
    return items


def score(items, preds, thr: float):
    tp = fp = fn = 0
    frame_hit = 0
    n_with_gt = 0
    by_cam = defaultdict(lambda: {"tp": 0, "fp": 0, "fn": 0, "frame_hit": 0, "n_frames": 0})
    for item in items:
        gt = item["gt"]
        pred = [(b, c) for b, c, _ in preds[item["image"]] if c >= thr]
        tpi, fpi, fni = match_preds(gt, pred)
        tp += tpi
        fp += fpi
        fn += fni
        cam = by_cam[item["camera"]]
        cam["tp"] += tpi
        cam["fp"] += fpi
        cam["fn"] += fni
        cam["n_frames"] += 1
        if gt:
            n_with_gt += 1
            if tpi > 0:
                frame_hit += 1
                cam["frame_hit"] += 1
    p, r = pr(tp, fp, fn)
    acc = tp / (tp + fp + fn) if (tp + fp + fn) else 0.0
    out = {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "precision": p,
        "recall": r,
        "box_accuracy_jaccard": acc,
        "n_gt_boxes": tp + fn,
        "n_pred_boxes": tp + fp,
        "frame_recall": frame_hit / n_with_gt if n_with_gt else 0.0,
        "frames_with_gt": n_with_gt,
        "frames_with_hit": frame_hit,
        "by_camera": {},
    }
    for name, row in by_cam.items():
        cp, cr = pr(row["tp"], row["fp"], row["fn"])
        out["by_camera"][name] = {
            **row,
            "precision": cp,
            "recall": cr,
            "frame_recall": row["frame_hit"] / row["n_frames"] if row["n_frames"] else 0.0,
        }
    return out


def main() -> int:
    items = load_items(GOLD)
    print(f"checkpoint: {CKPT}")
    print(f"gold frames: {len(items)}")
    model = load_ball_model(str(CKPT))
    preds = {}
    for item in items:
        frame = cv2.imread(str(item["path"]))
        if frame is None:
            raise RuntimeError(item["path"])
        dets = detect_balls(model, frame, 0.30, False)
        preds[item["image"]] = [
            (list(d.bbox), float(d.confidence), min(d.bbox[2], d.bbox[3]))
            for d in dets
        ]
        top = max((c for _, c, _ in preds[item["image"]]), default=0.0)
        print(f"{item['camera']} {item['image']} n={len(preds[item['image']])} top={top:.3f}")
    report = {
        "checkpoint": "models/v10_snaps/post_train/checkpoint.pth",
        "gold": "match2_gold_frames",
        "note": (
            "No 0.80 emit gate. Recall = GT balls found (IoU 0.5). "
            "Precision = published boxes that match GT. "
            "Gold 50 is held-out harvest of large/clear balls, not every ball in the match. "
            "Cam5plus has no gold labels. Train100 not used."
        ),
        "thresholds": {
            "0.3": score(items, preds, 0.30),
            "0.5": score(items, preds, 0.50),
        },
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(report, indent=2))
    for thr, block in report["thresholds"].items():
        print(f"\n=== conf >= {thr} ===")
        print(
            f"recall={block['recall']:.3f} precision={block['precision']:.3f} "
            f"jaccard={block['box_accuracy_jaccard']:.3f} "
            f"frame_recall={block['frame_recall']:.3f} "
            f"tp/fp/fn={block['tp']}/{block['fp']}/{block['fn']}"
        )
        for cam, row in block["by_camera"].items():
            print(
                f"  {cam}: R={row['recall']:.3f} P={row['precision']:.3f} "
                f"frame_R={row['frame_recall']:.3f} "
                f"tp/fp/fn={row['tp']}/{row['fp']}/{row['fn']}"
            )
    print(f"\nWrote {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
