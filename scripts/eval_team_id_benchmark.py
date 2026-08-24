#!/usr/bin/env python3
"""Evaluate team ID against benchmark JSON (GS-HOTA-lite metrics)."""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
_GS = ROOT / "scripts" / "gold_set"
sys.path.insert(0, str(_GS))

from raw_cam_id import load_match_raw  # noqa: E402

from src.perception.team_core import (  # noqa: E402
    TEAM_ASSIGN_CONF,
    assign_feature,
    fit_match_centroids,
    jersey_feature,
    torso_crop,
)
from src.review.team_live import TeamSession, label_player_pts  # noqa: E402

BENCH = ROOT / "data/gold/team_id_match3_500.json"
CROP_DIR = ROOT / "data/gold/team_id_match3_500_crops"
REPORT = ROOT / "reports/eval_match3/team_id_benchmark/scores.json"
RAW = ROOT / "data/raw/Match 3"


def _load_cfg() -> dict:
    import yaml

    return yaml.safe_load((ROOT / "configs/default.yaml").read_text(encoding="utf-8"))


def bootstrap_labels(data: dict) -> None:
    """High-confidence kit fractions → provisional 0/1 labels for unset instances."""
    for inst in data["instances"]:
        if inst.get("label") != -2:
            continue
        crop_path = CROP_DIR / inst["crop"]
        if not crop_path.is_file():
            continue
        crop = cv2.imread(str(crop_path))
        feat = jersey_feature(crop)
        if feat is None:
            inst["label"] = "invalid"
            inst["label_name"] = "invalid_crop"
            continue
        blue, white, yellow = float(feat[0]), float(feat[1]), float(feat[2])
        light = white + yellow
        if blue >= 0.42 and blue >= light + 0.15:
            inst["label"] = 0
            inst["label_name"] = "team_0_blue"
        elif light >= 0.42 and light >= blue + 0.15:
            inst["label"] = 1
            inst["label_name"] = "team_1_white_yellow"
        else:
            inst["label"] = -1
            inst["label_name"] = "unsure_gray"


def eval_crop_model(data: dict) -> dict:
    feats = []
    labs = []
    ids = []
    for inst in data["instances"]:
        lab = inst.get("label")
        if lab not in (0, 1):
            continue
        crop = cv2.imread(str(CROP_DIR / inst["crop"]))
        feat = jersey_feature(crop)
        if feat is None:
            continue
        feats.append(feat)
        labs.append(int(lab))
        ids.append(int(inst["id"]))
    fit = fit_match_centroids(feats, min_crops=5)
    if fit is None:
        return {"precision": 0.0, "n": 0}
    cents, radius = fit
    correct = 0
    gray = 0
    wrong = 0
    for feat, gt, iid in zip(feats, labs, ids):
        pred, conf = assign_feature(feat, cents, radius)
        if pred < 0:
            gray += 1
        elif int(pred) == int(gt):
            correct += 1
        else:
            wrong += 1
    denom = max(correct + wrong, 1)
    return {
        "precision": correct / denom,
        "gray_frac": gray / max(len(feats), 1),
        "mislabel": wrong,
        "correct": correct,
        "n": len(feats),
    }


def eval_swap_rate(data: dict) -> float:
    """Proxy swap rate on consecutive benchmark frames per cam."""
    by_cam: dict[str, list] = defaultdict(list)
    for inst in data["instances"]:
        if inst.get("label") not in (0, 1):
            continue
        by_cam[inst["cam"]].append(inst)
    swaps = 0
    pairs = 0
    for cam, rows in by_cam.items():
        rows.sort(key=lambda r: int(r["frame"]))
        prev_lab = None
        for r in rows:
            lab = int(r["label"])
            if prev_lab is not None and lab != prev_lab:
                swaps += 1
            pairs += 1
            prev_lab = lab
    return swaps / max(pairs, 1)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bootstrap", action="store_true", help="Auto-label unset crops")
    parser.add_argument("--bench", type=Path, default=BENCH)
    args = parser.parse_args()

    if not args.bench.is_file():
        print(f"Missing benchmark {args.bench}; run build_team_id_benchmark.py first")
        return 1

    data = json.loads(args.bench.read_text(encoding="utf-8"))
    if args.bootstrap:
        bootstrap_labels(data)
        args.bench.write_text(json.dumps(data, indent=2), encoding="utf-8")

    crop_scores = eval_crop_model(data)
    swap = eval_swap_rate(data)
    targets = {
        "precision_min": 0.85,
        "swap_max": 0.05,
        "gray_max": 0.20,
    }
    pass_prec = crop_scores.get("precision", 0.0) >= targets["precision_min"]
    pass_swap = swap <= targets["swap_max"]
    pass_gray = crop_scores.get("gray_frac", 1.0) <= targets["gray_max"]

    report = {
        "ts": datetime.now(timezone.utc).isoformat(),
        "benchmark": str(args.bench),
        "n_instances": data.get("n_instances", 0),
        "crop_model": crop_scores,
        "swap_rate_proxy": swap,
        "targets": targets,
        "pass": {
            "precision": pass_prec,
            "swap_rate": pass_swap,
            "gray_frac": pass_gray,
            "all": pass_prec and pass_swap and pass_gray,
        },
        "phase4_note": "Bhattacharyya hist distance + Mahalanobis outlier active in team_core",
    }
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0 if report["pass"]["all"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
