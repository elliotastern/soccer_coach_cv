#!/usr/bin/env python3
"""Mirror touchline left↔right for P8/P9 landmark names; refit H (and P8 H_player)."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts/gold_set"))

from match3_landmarks import (  # noqa: E402
    CALIB_DIR,
    _fit_H,
    draw_overlay,
    save_clicks,
    STILL_DIR,
    write_manifest,
)
from pitch1 import pitch1_landmarks  # noqa: E402

CAMS = ("P8", "P9")
MIRROR_FLAG = "p8_p9_touchline_mirror_applied"


def already_mirrored(rec: dict) -> bool:
    return bool(rec.get(MIRROR_FLAG))


def mark_mirrored(path: Path) -> None:
    rec = json.loads(path.read_text(encoding="utf-8"))
    rec[MIRROR_FLAG] = True
    path.write_text(json.dumps(rec, indent=2), encoding="utf-8")


def mirror_touchline_name(name: str) -> str:
    """Pitch +y left ↔ −y right: swap near/far touchline tokens in landmark id."""
    if name == "halfway_near_touch":
        return "halfway_far_touch"
    if name == "halfway_far_touch":
        return "halfway_near_touch"
    if "_near_" in name:
        return name.replace("_near_", "_far_", 1)
    if "_far_" in name:
        return name.replace("_far_", "_near_", 1)
    if name.endswith("_near"):
        return name[:-5] + "_far"
    if name.endswith("_far"):
        return name[:-4] + "_near"
    return name


def refit_h_player(cam: str, rec: dict) -> None:
    names = [mirror_touchline_name(n) for n in rec.get("player_landmark_names") or []]
    imgs = rec.get("player_image_points") or []
    if len(names) < 4:
        return
    lm = pitch1_landmarks()
    pitch_pts = [lm[n]["xy"] for n in names]
    Hp, _ = _fit_H(imgs, pitch_pts)
    rec["H_player"] = Hp.tolist()
    rec["player_landmark_names"] = names
    rec["player_image_points"] = [[float(x), float(y)] for x, y in imgs]
    errs = []
    for n, (px, py) in zip(names, imgs):
        from src.mapping.match3_xy import apply_H

        xy = apply_H(Hp, px, py)
        tx, ty = lm[n]["xy"]
        errs.append(float(np.hypot(xy[0] - tx, xy[1] - ty)))
    rec["player_roundtrip_max_m"] = round(max(errs), 4)
    rec["player_h_note"] = "H_player refit after P8/P9 touchline mirror; ball keeps H"
    still = STILL_DIR / f"{cam}.jpg"
    img = cv2.imread(str(still))
    if img is not None:
        ov = draw_overlay(img, Hp, imgs, names)
        cv2.imwrite(str(CALIB_DIR / f"{cam}_manual_overlay_player.jpg"), ov)


def main() -> int:
    for cam in CAMS:
        path = CALIB_DIR / f"{cam}_manual.json"
        rec = json.loads(path.read_text(encoding="utf-8"))
        if already_mirrored(rec):
            print(f"{cam}: already touchline-mirrored — skip", flush=True)
            continue
        old_names = list(rec.get("landmark_names") or [])
        imgs = rec.get("image_points") or []
        new_names = [mirror_touchline_name(n) for n in old_names]
        order = rec.get("order") or "goal_right"
        save_clicks(cam, order, imgs, landmark_names=new_names, min_n=3)
        rec2 = json.loads(path.read_text(encoding="utf-8"))
        if cam == "P8" and rec.get("H_player"):
            refit_h_player(cam, rec2)
            rec2[MIRROR_FLAG] = True
            path.write_text(json.dumps(rec2, indent=2), encoding="utf-8")
        else:
            mark_mirrored(path)
        print(f"{cam}: touchline-mirrored {len(old_names)} landmarks", flush=True)
        for o, n in zip(old_names, new_names):
            if o != n:
                print(f"  {o} -> {n}", flush=True)
    write_manifest()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
