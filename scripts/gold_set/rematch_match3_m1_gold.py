#!/usr/bin/env python3
"""Rematch Match 3 M1 strip gold_xy from focus-cam gt boxes after human edits."""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.mapping.match3_xy import load_calib, map_ball_box  # noqa: E402

STRIP = ROOT / "data/processed/gold_sets/match3_quad_p10_31/labels.json"
WH = (1920, 1080)
CLEAR_SIDE = 25.0


def rematch_frame(seed: dict, calib: dict) -> dict:
    balls = seed.get("gt_balls") or []
    if not balls:
        seed["clear"] = False
        seed["empty"] = True
        seed["gold_xy"] = None
        seed["gold_support"] = None
        seed.pop("gold_xy_source", None)
        return seed
    b = balls[0]
    box = (float(b["x"]), float(b["y"]), float(b["w"]), float(b["h"]))
    side = min(box[2], box[3])
    conf = float(seed.get("seed_conf") or seed.get("human_conf") or 1.0)
    hit = map_ball_box(calib, box, conf, frame_wh=WH)
    seed["empty"] = False
    seed["clear"] = side >= CLEAR_SIDE
    seed["seed_side"] = side
    if hit is not None:
        seed["gold_xy"] = [hit["xy"][0], hit["xy"][1]]
        seed["gold_support"] = hit["support"]
        seed["gold_xy_source"] = calib.get("camera") or "focus"
    # Focus H may not span FOV (e.g. P9 midfield). Keep prior other-cam gold_xy.
    return seed


def rematch_labels(payload: dict) -> dict:
    focus = payload.get("focus_cam") or "P10"
    calib = load_calib(focus)
    if calib is None:
        raise RuntimeError(f"missing calib {focus}")
    n_clear = n_gold = 0
    for fr in payload["frames"]:
        seed = (fr.get("cams") or {}).get(focus) or {}
        rematch_frame(seed, calib)
        fr.setdefault("cams", {})[focus] = seed
        if seed.get("clear"):
            n_clear += 1
        if seed.get("gold_xy"):
            n_gold += 1
    payload["n_clear"] = n_clear
    payload["n_gold_xy"] = n_gold
    payload["focus_cam"] = focus
    return payload


def pack_dir_for(payload: dict) -> Path:
    pack = payload.get("pack")
    if not pack:
        raise ValueError("labels missing pack")
    path = ROOT / "data/processed/gold_sets" / pack
    if not path.is_dir():
        raise FileNotFoundError(path)
    return path


def main() -> int:
    path = Path(sys.argv[1]) if len(sys.argv) > 1 else STRIP
    payload = json.loads(path.read_text(encoding="utf-8"))
    rematch_labels(payload)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"rematched {path}: clear={payload['n_clear']} gold_xy={payload['n_gold_xy']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
