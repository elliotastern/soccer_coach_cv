#!/usr/bin/env python3
"""Phase 0: H-consistency baseline — landmark round-trip + multi-cam map span.

Kill criterion for agree-shrink: if median pairwise mapped-ball span on
true multi-cam frames is already ≤ ~2.5 m, H is not the bottleneck.
"""
from __future__ import annotations

import json
import math
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts" / "gold_set"))

from ab_match3_defish_detect import CAMS, WH  # noqa: E402
from eval_match2_top_left_multicam_baseline import cache_load  # noqa: E402
from multicam_select_policy import MATCH3_THR_BY_CAM, filter_active  # noqa: E402
from src.mapping.match3_xy import load_calib, map_ball_box  # noqa: E402

CALIB_DIR = ROOT / "reports/eval_match3/match3_pitch_calib"
HOLDOUT_DIR = ROOT / "reports/eval_match3/pitchmap_gallery_holdout" / "det_cache"
OUT_DIR = ROOT / "reports/eval_match3/improve_eng_loop/h_consistency"
SPAN_KILL_M = 2.5
L1_MAX_RT_M = 0.15


def landmark_roundtrip_m(rec: dict) -> dict | None:
    img = rec.get("image_points") or []
    pitch = rec.get("pitch_points") or []
    H = rec.get("H") or rec.get("homography")
    if H is None or len(img) < 3 or len(img) != len(pitch):
        return None
    H = np.asarray(H, dtype=float)
    errs = []
    for (u, v), (X, Y) in zip(img, pitch):
        p = H @ np.array([float(u), float(v), 1.0], dtype=float)
        if abs(p[2]) < 1e-9:
            continue
        xy = (p[0] / p[2], p[1] / p[2])
        errs.append(float(math.hypot(xy[0] - float(X), xy[1] - float(Y))))
    if not errs:
        return None
    return {
        "n_landmarks": len(errs),
        "rt_mean_m": round(float(np.mean(errs)), 4),
        "rt_max_m": round(float(np.max(errs)), 4),
        "rt_pass_l1": float(np.max(errs)) <= L1_MAX_RT_M,
        "names": list(rec.get("landmark_names") or []),
        "has_hull": bool(rec.get("hull_image_points")),
        "projection3d_max_reproj_px": rec.get("projection3d_max_reproj_px"),
    }


def pairwise_spans_m(xys: list[tuple[float, float]]) -> list[float]:
    spans = []
    for i in range(len(xys)):
        for j in range(i + 1, len(xys)):
            spans.append(
                float(
                    math.hypot(xys[i][0] - xys[j][0], xys[i][1] - xys[j][1])
                )
            )
    return spans


def score_holdout_spans(calibs: dict) -> dict:
    all_spans: list[float] = []
    n_multi = 0
    n_frames = 0
    per_cache = []
    for path in sorted(HOLDOUT_DIR.glob("det_cache_*.json")):
        dets = cache_load(path)
        cams = [c for c in CAMS if c in dets]
        n = len(next(iter(dets.values())))
        cache_spans: list[float] = []
        multi = 0
        for i in range(n):
            n_frames += 1
            active = filter_active(dets, i, cams, MATCH3_THR_BY_CAM)
            xys = []
            for cam, rows in active.items():
                rec = calibs.get(cam)
                if rec is None:
                    continue
                box, conf, _ = rows[0]
                hit = map_ball_box(rec, box, float(conf), frame_wh=WH)
                if hit is not None:
                    xys.append((float(hit["xy"][0]), float(hit["xy"][1])))
            if len(xys) < 2:
                continue
            multi += 1
            n_multi += 1
            spans = pairwise_spans_m(xys)
            cache_spans.extend(spans)
            all_spans.extend(spans)
        row = {
            "cache": path.name,
            "n_multi_cam_frames": multi,
            "n_pair_spans": len(cache_spans),
            "span_median_m": None
            if not cache_spans
            else round(float(np.median(cache_spans)), 4),
            "span_p90_m": None
            if not cache_spans
            else round(float(np.percentile(cache_spans, 90)), 4),
        }
        per_cache.append(row)
    med = None if not all_spans else float(np.median(all_spans))
    return {
        "n_frames": n_frames,
        "n_multi_cam_frames": n_multi,
        "n_pair_spans": len(all_spans),
        "span_median_m": None if med is None else round(med, 4),
        "span_mean_m": None
        if not all_spans
        else round(float(np.mean(all_spans)), 4),
        "span_p90_m": None
        if not all_spans
        else round(float(np.percentile(all_spans, 90)), 4),
        "span_max_m": None
        if not all_spans
        else round(float(np.max(all_spans)), 4),
        "h_not_bottleneck": med is not None and med <= SPAN_KILL_M,
        "caches": per_cache,
    }


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    landmark = {}
    calibs = {}
    for path in sorted(CALIB_DIR.glob("*_manual.json")):
        if path.name.startswith("._"):
            continue
        cam = path.name.replace("_manual.json", "")
        rec = json.loads(path.read_text(encoding="utf-8"))
        calibs[cam] = rec
        rt = landmark_roundtrip_m(rec)
        if rt is not None:
            landmark[cam] = rt
    spans = score_holdout_spans(calibs)
    payload = {
        "ts": datetime.now(timezone.utc).isoformat(),
        "span_kill_m": SPAN_KILL_M,
        "l1_max_rt_m": L1_MAX_RT_M,
        "landmark_roundtrip": landmark,
        "holdout_pairwise_map_span": spans,
        "verdict": (
            "H not bottleneck — chase detect/recall before shrinking AGREE_M"
            if spans.get("h_not_bottleneck")
            else "H may contribute to multi-cam span — improve calib before AGREE_M shrink"
        ),
    }
    out = OUT_DIR / "h_consistency_baseline.json"
    out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps({
        "span_median_m": spans.get("span_median_m"),
        "span_p90_m": spans.get("span_p90_m"),
        "h_not_bottleneck": spans.get("h_not_bottleneck"),
        "landmark_cams": list(landmark.keys()),
        "wrote": str(out),
        "verdict": payload["verdict"],
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
