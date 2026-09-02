#!/usr/bin/env python3
"""Refit Match 3 H from existing landmark clicks (Pitch 1 meters).

Improves H consistency without new clicks: re-runs DLT from image_points /
pitch_points already in ``*_manual.json``, keeps hull/undistort/meta, writes
optional backup, and refreshes projection3d when possible.

Also exposes ``fit_h_from_paired_points`` for commercial-safe auto-keypoint
pipelines that emit the same (image, pitch) pairs — write via
``write_calib_h`` into the product calib JSON shape.
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts" / "gold_set"))

from report_match3_h_consistency import landmark_roundtrip_m  # noqa: E402

CALIB_DIR = ROOT / "reports/eval_match3/match3_pitch_calib"
OUT = ROOT / "reports/eval_match3/improve_eng_loop/h_consistency/h_refit_report.json"


def fit_h_from_paired_points(
    image_points: list,
    pitch_points: list,
    *,
    method: int = 0,
    ransac_reproj_threshold: float = 3.0,
) -> np.ndarray:
    """Pitch-1-safe H from ≥4 paired points (same shape product mapper loads).

    ``method=0`` is product DLT (exact). Pass ``cv2.RANSAC`` only for
    auto-keypoint A/B — never promote RANSAC H that worsens landmark RT.
    """
    src = np.float32(image_points)
    dst = np.float32(pitch_points)
    if len(src) < 4:
        raise ValueError(f"need ≥4 points, got {len(src)}")
    if len(src) != len(dst):
        raise ValueError("image_points / pitch_points length mismatch")
    kwargs = {"method": int(method)}
    if int(method) == int(cv2.RANSAC):
        kwargs["ransacReprojThreshold"] = float(ransac_reproj_threshold)
    H, _ = cv2.findHomography(src, dst, **kwargs)
    if H is None:
        raise RuntimeError("findHomography failed")
    return np.asarray(H, dtype=float)


def write_calib_h(
    path: Path,
    H: np.ndarray,
    *,
    backup: bool = True,
    source: str = "refit_from_landmarks",
) -> dict:
    """Update H/homography on an existing calib JSON; preserve other fields."""
    rec = json.loads(path.read_text(encoding="utf-8"))
    if backup:
        stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        bak = path.with_name(f"{path.stem}_pre_refit_{stamp}.json")
        shutil.copy2(path, bak)
        rec["_refit_backup"] = str(bak.relative_to(ROOT)) if bak.is_relative_to(ROOT) else str(bak)
    before = landmark_roundtrip_m(rec)
    rec["H"] = H.tolist()
    rec["homography"] = H.tolist()
    rec["h_source"] = source
    rec["h_refit_ts"] = datetime.now(timezone.utc).isoformat()
    after = landmark_roundtrip_m(rec)
    path.write_text(json.dumps(rec, indent=2), encoding="utf-8")
    cam = path.name.replace("_manual.json", "")
    try:
        from src.mapping.camera_projection import write_projection_to_calib

        write_projection_to_calib(cam, dry_run=False)
        rec = json.loads(path.read_text(encoding="utf-8"))
        after = landmark_roundtrip_m(rec)
    except Exception as exc:  # noqa: BLE001 — optional enrich
        rec["_projection3d_note"] = f"skip: {exc}"
        path.write_text(json.dumps(rec, indent=2), encoding="utf-8")
    return {"before": before, "after": after, "path": str(path)}


def refit_cam(path: Path, *, dry_run: bool, backup: bool) -> dict:
    rec = json.loads(path.read_text(encoding="utf-8"))
    img = rec.get("image_points") or []
    pitch = rec.get("pitch_points") or []
    cam = path.name.replace("_manual.json", "")
    if len(img) < 4:
        return {"cam": cam, "skipped": True, "reason": f"n_points={len(img)} < 4"}
    before = landmark_roundtrip_m(rec)
    H = fit_h_from_paired_points(img, pitch)
    if dry_run:
        trial = dict(rec)
        trial["H"] = H.tolist()
        after = landmark_roundtrip_m(trial)
        return {
            "cam": cam,
            "dry_run": True,
            "before": before,
            "after": after,
            "improved_max_rt": (
                before is not None
                and after is not None
                and after["rt_max_m"] <= before["rt_max_m"] + 1e-9
            ),
        }
    result = write_calib_h(path, H, backup=backup, source="refit_from_landmarks")
    result["cam"] = cam
    result["dry_run"] = False
    return result


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--cams",
        default="all",
        help="Comma cams or 'all' (default all manual calibs)",
    )
    p.add_argument(
        "--write",
        action="store_true",
        help="Write refit H into calib JSON (default: dry-run only)",
    )
    p.add_argument(
        "--no-backup",
        action="store_true",
        help="Skip timestamp backup when --write",
    )
    return p.parse_args()


def main() -> int:
    args = parse_args()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    if args.cams.strip() == "all":
        paths = [
            p
            for p in sorted(CALIB_DIR.glob("*_manual.json"))
            if not p.name.startswith("._")
        ]
    else:
        paths = [CALIB_DIR / f"{c.strip()}_manual.json" for c in args.cams.split(",")]
    rows = []
    for path in paths:
        if not path.is_file():
            rows.append({"cam": path.name, "skipped": True, "reason": "missing"})
            continue
        rows.append(refit_cam(path, dry_run=not args.write, backup=not args.no_backup))
    payload = {
        "ts": datetime.now(timezone.utc).isoformat(),
        "write": bool(args.write),
        "cams": rows,
        "note": (
            "Auto-keypoint path: detect Pitch 1 landmarks → fit_h_from_paired_points "
            "→ write_calib_h. Match 4 uses same P-code calib files."
        ),
    }
    OUT.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    for r in rows:
        b = (r.get("before") or {}).get("rt_max_m")
        a = (r.get("after") or {}).get("rt_max_m")
        print(f"{r.get('cam')}: rt_max {b} → {a} skipped={r.get('skipped', False)}")
    print(f"wrote {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
