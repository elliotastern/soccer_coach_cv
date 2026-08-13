#!/usr/bin/env python3
"""PoC ball metrics: P_emit @0.8 + clear-ball recall (min side >= 25px)."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.perception.rfdetr_local import load_ball_model

IOU_THR = 0.5
CLEAR_MIN_SIDE = 25.0


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--gold-dir", type=Path, default=ROOT / "data/processed/gold_sets/match1_1_100")
    p.add_argument(
        "--ball-checkpoint",
        type=Path,
        default=ROOT / "models/v6_snaps/epoch_110/checkpoint_best_regular.pth",
    )
    p.add_argument("--strip-max", type=int, default=49, help="Gold strip frames 0..N")
    p.add_argument(
        "--require-ball-gt",
        action="store_true",
        help="Only eval frames with >=1 ball GT (selection slice)",
    )
    p.add_argument("--min-thr", type=float, default=0.3)
    p.add_argument("--clear-min-side", type=float, default=CLEAR_MIN_SIDE)
    p.add_argument("--use-sahi", action="store_true")
    p.add_argument("--out", type=Path, default=None)
    return p.parse_args()


def iou_xywh(a, b) -> float:
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    ix1, iy1 = max(ax, bx), max(ay, by)
    ix2, iy2 = min(ax + aw, bx + bw), min(ay + ah, by + bh)
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    union = aw * ah + bw * bh - inter
    return inter / union if union > 0 else 0.0


def match_preds(gt_boxes, pred_boxes):
    matched = set()
    tp = 0
    for box, _ in sorted(pred_boxes, key=lambda x: -x[1]):
        best_j, best = -1, 0.0
        for j, g in enumerate(gt_boxes):
            if j in matched:
                continue
            v = iou_xywh(box, g)
            if v > best:
                best, best_j = v, j
        if best >= IOU_THR and best_j >= 0:
            tp += 1
            matched.add(best_j)
    fp = len(pred_boxes) - tp
    fn = len(gt_boxes) - len(matched)
    return tp, fp, fn


def pr(tp, fp, fn):
    p = tp / (tp + fp) if (tp + fp) else 0.0
    r = tp / (tp + fn) if (tp + fn) else 0.0
    return p, r


def detect_balls(model, frame_bgr, thr: float, use_sahi: bool):
    from src.perception.ball_prelabel import BallPrelabelConfig, BallPrelabeler

    pre = BallPrelabeler(
        model,
        BallPrelabelConfig(
            threshold=thr,
            use_sahi=use_sahi,
            sahi_fallback_only=True,
            sahi_recover_only=True,
            use_size_filter=True,
            topk=2,
            use_kalman=False,
            min_side=4,
            max_side=240,
        ),
    )
    return pre.detect_bgr(frame_bgr)


def main():
    args = parse_args()
    coco = json.loads((args.gold_dir / "gold" / "annotations.coco.json").read_text())
    cats = {c["id"]: c["name"] for c in coco["categories"]}
    # map file_name -> strip index from manifest if present
    manifest = {}
    man_path = args.gold_dir / "manifest.json"
    if man_path.is_file():
        man = json.loads(man_path.read_text())
        for fr in man.get("frames", []):
            manifest[fr["image"]] = fr.get("strip_frame")

    ball_anns = [a for a in coco["annotations"] if cats.get(a["category_id"]) == "ball"]
    by_image = {}
    for a in ball_anns:
        by_image.setdefault(a["image_id"], []).append(a["bbox"])

    images = []
    for im in coco["images"]:
        strip = manifest.get(im["file_name"])
        if strip is None:
            # try parse cam*_f*
            strip = 0
        if strip is not None and int(strip) > args.strip_max:
            continue
        images.append(im)

    # Prefer strip from manifest only
    if manifest:
        images = []
        id_by_name = {im["file_name"]: im for im in coco["images"]}
        for name, strip in manifest.items():
            if strip is None or int(strip) > args.strip_max:
                continue
            if name in id_by_name:
                images.append(id_by_name[name])

    if args.require_ball_gt:
        images = [im for im in images if by_image.get(im["id"])]

    print(f"checkpoint: {args.ball_checkpoint}")
    print(
        f"images in strip 0-{args.strip_max}"
        f"{' (ball-present only)' if args.require_ball_gt else ''}: {len(images)}"
    )
    model = load_ball_model(str(args.ball_checkpoint))

    preds = {}
    for im in images:
        path = args.gold_dir / "images" / im["file_name"]
        frame = cv2.imread(str(path))
        if frame is None:
            raise RuntimeError(f"missing {path}")
        dets = detect_balls(model, frame, args.min_thr, args.use_sahi)
        preds[im["id"]] = [
            (list(d.bbox), float(d.confidence), min(d.bbox[2], d.bbox[3]))
            for d in dets
        ]
        top = max((c for _, c, _ in preds[im["id"]]), default=0.0)
        print(f"{im['file_name']}: n={len(preds[im['id']])} top={top:.3f}")

    try:
        ckpt_rel = str(args.ball_checkpoint.resolve().relative_to(ROOT))
    except ValueError:
        ckpt_rel = str(args.ball_checkpoint)
    report = {
        "checkpoint": ckpt_rel,
        "strip": f"0-{args.strip_max}",
        "require_ball_gt": bool(args.require_ball_gt),
        "n_images": len(images),
        "clear_min_side": args.clear_min_side,
        "use_sahi": args.use_sahi,
        "thresholds": {},
    }

    for thr in (0.5, 0.8):
        tp = fp = fn = 0
        clear_tp = clear_fn = 0
        n_clear_gt = 0
        for im in images:
            gt = by_image.get(im["id"], [])
            pred = [(b, c) for b, c, _ in preds[im["id"]] if c >= thr]
            tpi, fpi, fni = match_preds(gt, pred)
            tp += tpi
            fp += fpi
            fn += fni
            clear_gt = [g for g in gt if min(g[2], g[3]) >= args.clear_min_side]
            n_clear_gt += len(clear_gt)
            if clear_gt:
                ct, _, cf = match_preds(clear_gt, pred)
                clear_tp += ct
                clear_fn += cf
        p, r = pr(tp, fp, fn)
        clear_r = clear_tp / (clear_tp + clear_fn) if (clear_tp + clear_fn) else 0.0
        # P_emit: precision among emitted (undefined if zero emits → report null)
        p_emit = p if (tp + fp) > 0 else None
        block = {
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "precision": p,
            "recall": r,
            "P_emit": p_emit,
            "n_emitted": tp + fp,
            "clear_ball": {
                "n_gt": n_clear_gt,
                "tp": clear_tp,
                "fn": clear_fn,
                "recall": clear_r,
            },
            "poc_pass_P_emit": bool(p_emit is not None and p_emit >= 0.80),
        }
        report["thresholds"][str(thr)] = block
        print(f"\n=== conf >= {thr} ===")
        print(
            f"ball: P={p:.3f} R={r:.3f} P_emit={p_emit} "
            f"(tp={tp} fp={fp} fn={fn} emitted={tp+fp})"
        )
        print(
            f"clear-ball (side>={args.clear_min_side}): "
            f"R={clear_r:.3f} (tp={clear_tp} fn={clear_fn} n_gt={n_clear_gt})"
        )
        print(f"PoC P_emit>=0.80: {block['poc_pass_P_emit']}")

    if args.out:
        out = args.out
    else:
        tag = args.ball_checkpoint.stem
        suffix = "_sahi" if args.use_sahi else ""
        ball = "_ballgt" if args.require_ball_gt else ""
        out = ROOT / "reports" / f"poc_{tag}{ball}{suffix}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2))
    print(f"\nWrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
