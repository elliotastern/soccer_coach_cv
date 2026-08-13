#!/usr/bin/env python3
"""Ablate v10 inference post-process on Match2 train 87, then held-out gold 50.

Pick on train @0.3 (precision >= 0.90, no extra FPs vs baseline). Gold XML is source of truth.
Never trains. Never uses gold for picking.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import cv2

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts" / "gold_set"))

from eval_match2_v10_video_system import load_gold_items
from eval_poc_ball_metrics import match_preds, pr
from src.perception.ball_prelabel import (
    BallPrelabelConfig,
    BallPrelabeler,
    filter_ball_geometry,
    nms_balls,
    topk_balls,
)
from src.perception.rfdetr_local import load_ball_model
from src.state.types import Detection

CKPT = ROOT / "models/v10_snaps/post_train/checkpoint.pth"
TRAIN_PACK = ROOT / "data/processed/gold_sets/match2_train_label100"
TRAIN_COCO = ROOT / "data/processed/gold_sets/match2_train_test/train/_annotations.coco.json"
GOLD_DIR = ROOT / "data/processed/gold_sets/match2_gold_frames"
OUT = ROOT / "reports/eval_match2_v10/postprocess_ablation.json"
PREC_FLOOR = 0.90
BASELINE = "baseline_thr30_size_topk2"
SIZE = dict(use_size_filter=True, min_side=4, max_side=240, use_kalman=False)


def load_train_items():
    names = {
        im["file_name"]
        for im in json.loads(TRAIN_COCO.read_text())["images"]
    }
    items = load_gold_items(TRAIN_PACK)
    out = [it for it in items if it["image"] in names and it["gt"]]
    if len(out) != 87:
        raise RuntimeError(f"expected 87 train items with GT, got {len(out)}")
    return out


def unflip_bbox(bbox, width: float):
    x, y, w, h = bbox
    return (width - x - w, y, w, h)


def unflip_det(det: Detection, width: float) -> Detection:
    return Detection(
        class_id=det.class_id,
        confidence=det.confidence,
        bbox=unflip_bbox(det.bbox, width),
        class_name=det.class_name,
    )


def dets_as_rows(dets):
    return [
        (list(d.bbox), float(d.confidence), min(d.bbox[2], d.bbox[3]))
        for d in dets
    ]


def score_preds(items, preds, thr: float):
    tp = fp = fn = 0
    for item in items:
        pred = [(b, c) for b, c, _ in preds[item["image"]] if c >= thr]
        tpi, fpi, fni = match_preds(item["gt"], pred)
        tp += tpi
        fp += fpi
        fn += fni
    p, r = pr(tp, fp, fn)
    n_emitted = tp + fp
    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "precision": p,
        "recall": r,
        "P_emit": p if n_emitted > 0 else None,
        "n_emitted": n_emitted,
    }


def pick_winner(rows, baseline_name: str = BASELINE):
    base = next(r for r in rows if r["name"] == baseline_name)
    base_fp = base["train"]["0.3"]["fp"]
    eligible = []
    for row in rows:
        m = row["train"]["0.3"]
        if m["precision"] < PREC_FLOOR:
            continue
        if m["fp"] > base_fp:
            continue
        eligible.append(row)
    if not eligible:
        return base
    eligible.sort(key=lambda r: (-r["train"]["0.3"]["recall"], -r["train"]["0.3"]["precision"]))
    return eligible[0]


def technique_specs():
    return [
        {
            "name": BASELINE,
            "kind": "prelabel",
            "cfg": BallPrelabelConfig(threshold=0.30, use_sahi=False, topk=2, **SIZE),
        },
        {
            "name": "topk1",
            "kind": "prelabel",
            "cfg": BallPrelabelConfig(threshold=0.30, use_sahi=False, topk=1, **SIZE),
        },
        {
            "name": "topk3",
            "kind": "prelabel",
            "cfg": BallPrelabelConfig(threshold=0.30, use_sahi=False, topk=3, **SIZE),
        },
        {
            "name": "multiscale_1p5",
            "kind": "prelabel",
            "cfg": BallPrelabelConfig(
                threshold=0.30, use_sahi=False, use_multiscale=True, topk=2, **SIZE
            ),
        },
        {
            "name": "sahi_fallback_only",
            "kind": "prelabel",
            "cfg": BallPrelabelConfig(
                threshold=0.30,
                use_sahi=True,
                sahi_fallback_only=True,
                sahi_recover_only=True,
                topk=2,
                **SIZE,
            ),
        },
        {
            "name": "sahi_recover_always",
            "kind": "prelabel",
            "cfg": BallPrelabelConfig(
                threshold=0.30,
                use_sahi=True,
                sahi_fallback_only=False,
                sahi_recover_only=True,
                topk=2,
                **SIZE,
            ),
        },
        {
            "name": "hflip_tta_nms",
            "kind": "tta",
            "cfg": BallPrelabelConfig(threshold=0.30, use_sahi=False, topk=99, **SIZE),
        },
    ]


def detect_hflip_tta(pre, frame_bgr):
    dets = pre.detect_bgr(frame_bgr)
    flipped = cv2.flip(frame_bgr, 1)
    width = float(frame_bgr.shape[1])
    flipped_dets = [unflip_det(d, width) for d in pre.detect_bgr(flipped)]
    merged = nms_balls(list(dets) + flipped_dets, iou_thr=0.4)
    merged = filter_ball_geometry(
        merged, min_side=4, max_side=240, image_width=width
    )
    return topk_balls(merged, k=2)


def run_spec(model, items, spec, tag: str):
    pre = BallPrelabeler(model, spec["cfg"])
    preds = {}
    for i, item in enumerate(items):
        frame = cv2.imread(str(item["image_path"]))
        if frame is None:
            raise RuntimeError(f"missing jpeg {item['image_path']}")
        if spec["kind"] == "tta":
            dets = detect_hflip_tta(pre, frame)
        else:
            dets = pre.detect_bgr(frame)
        preds[item["image"]] = dets_as_rows(dets)
        top = max((c for _, c, _ in preds[item["image"]]), default=0.0)
        print(
            f"{tag} {spec['name']} {i:02d}/{len(items)-1} "
            f"{item['image']} n={len(preds[item['image']])} top={top:.3f}",
            flush=True,
        )
    return preds


def cfg_public(spec):
    cfg = spec["cfg"]
    return {
        "kind": spec["kind"],
        "threshold": cfg.threshold,
        "topk": cfg.topk,
        "use_sahi": cfg.use_sahi,
        "sahi_fallback_only": cfg.sahi_fallback_only,
        "sahi_recover_only": cfg.sahi_recover_only,
        "use_multiscale": cfg.use_multiscale,
        "use_kalman": cfg.use_kalman,
    }


def main() -> int:
    train_items = load_train_items()
    gold_items = load_gold_items(GOLD_DIR)
    print(f"train={len(train_items)} gold={len(gold_items)}", flush=True)
    model = load_ball_model(str(CKPT))
    rows = []
    for spec in technique_specs():
        train_preds = run_spec(model, train_items, spec, "train")
        train_m = {
            "0.3": score_preds(train_items, train_preds, 0.30),
            "0.8": score_preds(train_items, train_preds, 0.80),
        }
        print(
            f"TRAIN {spec['name']} @0.3 R={train_m['0.3']['recall']:.3f} "
            f"P={train_m['0.3']['precision']:.3f} fp={train_m['0.3']['fp']} | "
            f"@0.8 P_emit={train_m['0.8']['P_emit']} n={train_m['0.8']['n_emitted']}",
            flush=True,
        )
        rows.append({"name": spec["name"], "cfg": cfg_public(spec), "train": train_m, "gold": None})

    winner = pick_winner(rows)
    print(f"PICKED on train: {winner['name']}", flush=True)

    for spec, row in zip(technique_specs(), rows):
        gold_preds = run_spec(model, gold_items, spec, "gold")
        row["gold"] = {
            "0.3": score_preds(gold_items, gold_preds, 0.30),
            "0.8": score_preds(gold_items, gold_preds, 0.80),
        }
        g = row["gold"]
        print(
            f"GOLD {spec['name']} @0.3 R={g['0.3']['recall']:.3f} "
            f"P={g['0.3']['precision']:.3f} fp={g['0.3']['fp']} | "
            f"@0.8 P_emit={g['0.8']['P_emit']} n={g['0.8']['n_emitted']}",
            flush=True,
        )

    report = {
        "checkpoint": "models/v10_snaps/post_train/checkpoint.pth",
        "tune_on": "match2_train_label100 train split (87), not gold",
        "picked": winner["name"],
        "baseline": BASELINE,
        "all": rows,
        "cited_not_rerun": {
            "kalman": "reports/eval_match2_v10/track_tune.md — gold R=0.71 P=0.88",
            "bytetrack": "reports/eval_match2_v10/track_tune.md — gold R=0.90, emit0.8 R=0.40",
        },
        "read": (
            "Pick used train recall @0.3 with precision >= 0.90 and fp <= baseline. "
            "Gold scored after pick. Kalman/ByteTrack not re-run."
        ),
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(report, indent=2))
    print(f"Wrote {OUT}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
