#!/usr/bin/env python3
"""Per-cam + selection-method P/R from multicam eval_pack labels.json."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_LABELS = ROOT / "data/processed/multicam_20s_match1/eval_pack/labels.json"
CAMS = ["cam8", "cam9", "cam11", "cam13"]
IOU_THR = 0.5


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--labels", type=Path, default=DEFAULT_LABELS)
    p.add_argument("--thresholds", nargs="+", type=float, default=[0.3, 0.5, 0.8])
    return p.parse_args()


def iou_xywh(a, b) -> float:
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    ax2, ay2 = ax + aw, ay + ah
    bx2, by2 = bx + bw, by + bh
    ix1, iy1 = max(ax, bx), max(ay, by)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    union = aw * ah + bw * bh - inter
    return inter / union if union > 0 else 0.0


def as_xywh(b) -> list[float]:
    if isinstance(b, dict):
        if "bbox" in b:
            return [float(x) for x in b["bbox"]]
        return [float(b["x"]), float(b["y"]), float(b["w"]), float(b["h"])]
    return [float(x) for x in b]


def gt_boxes(item: dict) -> list[list[float]] | None:
    """None = unlabeled; [] = confirmed empty."""
    if item.get("empty") is True:
        return []
    if item.get("gt_balls") is None:
        return None
    return [as_xywh(b) for b in item["gt_balls"]]


def pred_boxes(item: dict, thr: float) -> list[tuple[list[float], float, float]]:
    out = []
    for b in item.get("prelabel_balls") or []:
        conf = float(b.get("confidence", 0))
        if conf < thr:
            continue
        box = as_xywh(b)
        side = min(box[2], box[3])
        out.append((box, conf, side))
    out.sort(key=lambda x: -x[1])
    return out


def match_tp(gt: list[list[float]], preds: list[tuple[list[float], float, float]]):
    """Greedy one-to-one. Returns tp, fp, fn."""
    matched = set()
    tp = 0
    for box, _, _ in preds:
        best_j, best = -1, 0.0
        for j, g in enumerate(gt):
            if j in matched:
                continue
            v = iou_xywh(box, g)
            if v > best:
                best, best_j = v, j
        if best >= IOU_THR and best_j >= 0:
            tp += 1
            matched.add(best_j)
    fp = len(preds) - tp
    fn = len(gt) - len(matched)
    return tp, fp, fn


def pr(tp, fp, fn):
    p = tp / (tp + fp) if (tp + fp) else 0.0
    r = tp / (tp + fn) if (tp + fn) else 0.0
    return p, r


def labeled_timestamps(data: dict):
    """Yield ts only if ALL cams labeled (gt or empty)."""
    for ts in data["timestamps"]:
        if all(gt_boxes(ts["cams"][c]) is not None for c in CAMS):
            yield ts


def score_per_cam(timestamps, thr: float) -> dict:
    totals = {c: {"tp": 0, "fp": 0, "fn": 0} for c in CAMS}
    for ts in timestamps:
        for cam in CAMS:
            gt = gt_boxes(ts["cams"][cam])
            preds = pred_boxes(ts["cams"][cam], thr)
            tp, fp, fn = match_tp(gt, preds)
            totals[cam]["tp"] += tp
            totals[cam]["fp"] += fp
            totals[cam]["fn"] += fn
    out = {}
    for cam in CAMS:
        t = totals[cam]
        p, r = pr(t["tp"], t["fp"], t["fn"])
        out[cam] = {**t, "precision": p, "recall": r}
    return out


def pick_selected(ts, thr: float, mode: str):
    """Return (cam, pred_tuple) or (None, None)."""
    cands = []
    for cam in CAMS:
        preds = pred_boxes(ts["cams"][cam], thr)
        if not preds:
            continue
        box, conf, side = preds[0]
        if mode == "max_conf":
            score = conf
        elif mode == "size_weighted":
            score = conf * side
        elif mode == "nearest_side":
            score = side
        else:
            raise ValueError(mode)
        cands.append((score, cam, preds[0]))
    if not cands:
        return None, None
    cands.sort(key=lambda x: -x[0])
    return cands[0][1], cands[0][2]


def score_selection(timestamps, thr: float, mode: str) -> dict:
    """
    System metric in image space of the chosen cam:
    - TP if chosen det matches that cam's GT
    - FP if chosen det does not match (or GT empty on that cam)
    - FN if any cam has GT ball but no TP from the selection
      (missed the physical ball this timestamp)
    """
    tp = fp = fn = 0
    for ts in timestamps:
        has_ball = any(len(gt_boxes(ts["cams"][c])) > 0 for c in CAMS)
        cam, pred = pick_selected(ts, thr, mode)
        if pred is None:
            if has_ball:
                fn += 1
            continue
        gt = gt_boxes(ts["cams"][cam])
        tpi, fpi, _ = match_tp(gt, [pred])
        if tpi:
            tp += 1
        else:
            fp += 1
            if has_ball:
                # selected wrong cam/box while ball exists somewhere
                fn += 1
    p, r = pr(tp, fp, fn)
    return {"tp": tp, "fp": fp, "fn": fn, "precision": p, "recall": r}


def score_oracle(timestamps, thr: float) -> dict:
    """Upper bound: timestamp is TP if ANY cam has a TP det."""
    tp = fp = fn = 0
    for ts in timestamps:
        has_ball = any(len(gt_boxes(ts["cams"][c])) > 0 for c in CAMS)
        any_tp = False
        any_pred = False
        for cam in CAMS:
            preds = pred_boxes(ts["cams"][cam], thr)
            if preds:
                any_pred = True
            tpi, _, _ = match_tp(gt_boxes(ts["cams"][cam]), preds)
            if tpi:
                any_tp = True
        if any_tp:
            tp += 1
        elif has_ball:
            fn += 1
        elif any_pred:
            fp += 1
    p, r = pr(tp, fp, fn)
    return {"tp": tp, "fp": fp, "fn": fn, "precision": p, "recall": r}


def fmt(row: dict) -> str:
    return (
        f"P={row['precision']:.3f} R={row['recall']:.3f} "
        f"(tp={row['tp']} fp={row['fp']} fn={row['fn']})"
    )


def main():
    args = parse_args()
    if not args.labels.is_file():
        raise SystemExit(f"labels not found: {args.labels}")
    data = json.loads(args.labels.read_text())
    timestamps = list(labeled_timestamps(data))
    n_all = len(data["timestamps"])
    print(f"labels: {args.labels}")
    print(f"fully-labeled timestamps: {len(timestamps)}/{n_all}")
    print(f"IoU thr: {IOU_THR}")
    if not timestamps:
        print("\nNo complete labels yet.")
        print("1) Open eval_pack/labeler.html")
        print("2) Label all 4 cams at each timestamp (box or Empty)")
        print("3) Save labels.json into eval_pack/")
        print("4) Re-run this script")
        return 1

    report = {"n_timestamps": len(timestamps), "thresholds": {}}
    for thr in args.thresholds:
        print(f"\n=== conf >= {thr} ===")
        per = score_per_cam(timestamps, thr)
        print("Per camera:")
        for cam in CAMS:
            print(f"  {cam}: {fmt(per[cam])}")
        oracle = score_oracle(timestamps, thr)
        maxc = score_selection(timestamps, thr, "max_conf")
        sizew = score_selection(timestamps, thr, "size_weighted")
        near = score_selection(timestamps, thr, "nearest_side")
        print("Multi-cam selection:")
        print(f"  oracle_any_tp:   {fmt(oracle)}")
        print(f"  max_conf:        {fmt(maxc)}")
        print(f"  size_weighted:   {fmt(sizew)}")
        print(f"  nearest_side:    {fmt(near)}")
        report["thresholds"][str(thr)] = {
            "per_cam": per,
            "oracle_any_tp": oracle,
            "max_conf": maxc,
            "size_weighted": sizew,
            "nearest_side": near,
        }

    out = args.labels.parent / "metrics_pr.json"
    out.write_text(json.dumps(report, indent=2))
    print(f"\nWrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
