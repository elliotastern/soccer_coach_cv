#!/usr/bin/env python3
"""Match 2 auto landmark → homography for Cam4plus/Cam5plus (v2).

Loads multi-frame stills from 4quad sources, runs HomographyEstimator.estimate_averaged,
applies a hard overlay geometry gate, retries up to 3 configs per cam.
Writes reports/eval_match2_v10/match2_pitch_calib/*_auto_v2.json only when pass=true.
Never trains. Manual fallback if exhausted.
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.mapping.homography import HomographyEstimator, transform_point  # noqa: E402
from src.mapping.pitch_keypoint_detector import (  # noqa: E402
    detect_pitch_keypoints_auto_averaged,
)
from src.mapping.fov_aware_homography import estimate_H_center_averaged  # noqa: E402

OUT = ROOT / "reports/eval_match2_v10/match2_pitch_calib"
SRC = ROOT / "reports/eval_match2_v10/4quad_test/source"
CAMS = ["Cam4plus", "Cam5plus"]
STEMS = [
    "quad_top_left_t00026.0s",
    "quad_bottom_right_t00412.0s",
]
# Detect at 1280 for speed (4K sources); H is scaled back to native width.
DETECT_W = 1280
PL, PW = 105.0, 68.0

# Retry bundle (max 3). FOV-aware center-circle first (matches what side cams see).
ATTEMPTS = [
    {"stem": STEMS[0], "n_frames": 10, "step": 12, "correct_distortion": False,
     "tag": "tl_fov_center", "method": "fov_center"},
    {"stem": STEMS[1], "n_frames": 10, "step": 12, "correct_distortion": False,
     "tag": "br_fov_center", "method": "fov_center"},
    {"stem": STEMS[0], "n_frames": 12, "step": 10, "correct_distortion": False,
     "tag": "tl_dist_off", "method": "averaged"},
]


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", type=Path, default=OUT)
    p.add_argument("--cams", nargs="*", default=CAMS)
    return p.parse_args()


def load_frames(stem: str, cam: str, n_frames: int, step: int):
    """Return (detect_frames, scale native_w/detect_w, native_wh)."""
    path = SRC / f"{stem}_{cam}.mp4"
    if not path.is_file():
        raise FileNotFoundError(path)
    cap = cv2.VideoCapture(str(path))
    frames = []
    scale = 1.0
    native_wh = None
    i = 0
    while len(frames) < n_frames:
        ok, fr = cap.read()
        if not ok:
            break
        if i % step == 0:
            h, w = fr.shape[:2]
            native_wh = (w, h)
            if w != DETECT_W:
                scale = w / float(DETECT_W)
                fr = cv2.resize(fr, (DETECT_W, int(round(h * DETECT_W / w))))
            frames.append(fr)
        i += 1
    cap.release()
    return frames, scale, native_wh


def pitch_to_img(H: np.ndarray, x: float, y: float):
    Hi = np.linalg.inv(H)
    v = Hi @ np.array([x, y, 1.0], dtype=float)
    if abs(v[2]) < 1e-9:
        return None
    return float(v[0] / v[2]), float(v[1] / v[2])


def landmark_debug(frames: list) -> dict:
    try:
        data = detect_pitch_keypoints_auto_averaged(
            frames,
            pitch_length=PL,
            pitch_width=PW,
            min_points=4,
            max_points=40,
            min_frames=min(5, len(frames)),
        )
    except Exception as e:
        return {"ok": False, "error": f"{type(e).__name__}: {e}"}
    if data is None:
        return {"ok": False, "error": "no_keypoints"}
    types = Counter()
    kps = data.get("keypoints") or []
    for kp in kps:
        types[getattr(kp, "landmark_type", "?")] += 1
    return {
        "ok": True,
        "n_image_points": len(data.get("image_points") or []),
        "n_keypoints": len(kps),
        "types": dict(types),
    }


def overlay_gate(H: np.ndarray, frame_wh: tuple[int, int]) -> dict:
    """Hard geometry checks; rejects prior bad Cam4/Cam5 autos."""
    w, h = frame_wh
    corners_pitch = [
        (-PL / 2, -PW / 2),
        (PL / 2, -PW / 2),
        (PL / 2, PW / 2),
        (-PL / 2, PW / 2),
    ]
    corners = []
    for xy in corners_pitch:
        p = pitch_to_img(H, *xy)
        if p is None:
            return {"pass": False, "reason": "corner_project_failed", "corners": []}
        corners.append(p)
    xs = [c[0] for c in corners]
    ys = [c[1] for c in corners]
    span_x = max(xs) - min(xs)
    span_y = max(ys) - min(ys)
    near = sum(1 for x, y in corners if -0.35 * w <= x <= 1.35 * w and -0.35 * h <= y <= 1.35 * h)

    # Center + circle samples
    center = pitch_to_img(H, 0.0, 0.0)
    circ = []
    for a in np.linspace(0, 2 * np.pi, 24, endpoint=False):
        p = pitch_to_img(H, 9.15 * np.cos(a), 9.15 * np.sin(a))
        if p is not None:
            circ.append(p)
    circ_span = 0.0
    if circ:
        cxs = [p[0] for p in circ]
        cys = [p[1] for p in circ]
        circ_span = max(max(cxs) - min(cxs), max(cys) - min(cys))

    reasons = []
    if near < 3:
        reasons.append("corners_far_from_frame")
    if span_x < 0.25 * w:
        reasons.append("span_x_too_small")
    if span_y < 0.18 * h:
        reasons.append("span_y_too_small")  # rejects extreme Cam5 thin strip
    if span_y / max(span_x, 1.0) < 0.12:
        reasons.append("aspect_too_flat")
    # Axis-aligned full-frame trap (failed Cam4): nearly rectangular, parallel to axes, huge coverage
    top_y = sorted(ys)[:2]
    bot_y = sorted(ys)[-2:]
    left_x = sorted(xs)[:2]
    right_x = sorted(xs)[-2:]
    y_pair_gap = abs(np.mean(top_y) - np.mean(bot_y))
    x_pair_gap = abs(np.mean(left_x) - np.mean(right_x))
    top_flat = abs(top_y[0] - top_y[1]) < 0.04 * h
    bot_flat = abs(bot_y[0] - bot_y[1]) < 0.04 * h
    left_flat = abs(left_x[0] - left_x[1]) < 0.04 * w
    right_flat = abs(right_x[0] - right_x[1]) < 0.04 * w
    covers = span_x > 0.75 * w and span_y > 0.75 * h
    if covers and top_flat and bot_flat and left_flat and right_flat:
        reasons.append("axis_aligned_fullframe")
    # Thin horizontal band across full width (Cam5 false pass)
    if top_flat and bot_flat and span_x > 0.85 * w and span_y < 0.40 * h:
        reasons.append("thin_axis_aligned_band")
    if circ_span > 0 and circ_span < 0.03 * min(w, h):
        reasons.append("center_circle_degenerate")
    if circ_span > 1.2 * max(w, h):
        reasons.append("center_circle_exploded")

    ok = len(reasons) == 0
    return {
        "pass": ok,
        "reason": "ok" if ok else ",".join(reasons),
        "corners": [[round(x, 1), round(y, 1)] for x, y in corners],
        "span_px": [round(span_x, 1), round(span_y, 1)],
        "corners_near_frame": near,
        "center_img": None if center is None else [round(center[0], 1), round(center[1], 1)],
        "circ_span_px": round(circ_span, 1),
        "y_pair_gap": round(y_pair_gap, 1),
        "x_pair_gap": round(x_pair_gap, 1),
    }


def draw_overlay(frame: np.ndarray, H: np.ndarray, gate: dict) -> np.ndarray:
    vis = frame.copy()
    corners = gate.get("corners") or []
    if len(corners) == 4:
        pts = np.array([[int(round(x)), int(round(y))] for x, y in corners], np.int32)
        cv2.polylines(vis, [pts], True, (0, 255, 255), 3)
        for i, p in enumerate(pts):
            cv2.circle(vis, tuple(p), 8, (0, 0, 255), -1)
            cv2.putText(vis, f"C{i}", (p[0] + 6, p[1] - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
    circ = []
    for a in np.linspace(0, 2 * np.pi, 48, endpoint=False):
        p = pitch_to_img(H, 9.15 * np.cos(a), 9.15 * np.sin(a))
        if p is not None:
            circ.append([int(round(p[0])), int(round(p[1]))])
    if len(circ) >= 3:
        cv2.polylines(vis, [np.array(circ, np.int32)], True, (255, 0, 0), 2)
    mid_x = [pitch_to_img(H, 0, -PW / 2), pitch_to_img(H, 0, PW / 2)]
    mid_y = [pitch_to_img(H, -PL / 2, 0), pitch_to_img(H, PL / 2, 0)]
    if all(mid_x) and all(mid_y):
        cv2.line(vis, (int(mid_x[0][0]), int(mid_x[0][1])), (int(mid_x[1][0]), int(mid_x[1][1])), (0, 255, 0), 2)
        cv2.line(vis, (int(mid_y[0][0]), int(mid_y[0][1])), (int(mid_y[1][0]), int(mid_y[1][1])), (0, 255, 0), 2)
    color = (0, 200, 0) if gate.get("pass") else (0, 0, 255)
    tag = f"gate={'PASS' if gate.get('pass') else 'FAIL'}  {gate.get('reason', '')}"
    cv2.rectangle(vis, (8, 8), (min(vis.shape[1] - 8, 900), 56), (0, 0, 0), -1)
    cv2.putText(vis, tag[:90], (16, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
    return vis


def scale_H_to_native(H_detect: np.ndarray, scale: float) -> np.ndarray:
    """H_detect maps detect-res pixels → pitch; return H for native pixels."""
    if abs(scale - 1.0) < 1e-6:
        return H_detect
    S = np.array([[scale, 0.0, 0.0], [0.0, scale, 0.0], [0.0, 0.0, 1.0]], dtype=float)
    return H_detect @ np.linalg.inv(S)


def run_attempt(cam: str, attempt: dict) -> dict:
    stem = attempt["stem"]
    frames, scale, native_wh = load_frames(stem, cam, attempt["n_frames"], attempt["step"])
    method = attempt.get("method", "averaged")
    result = {
        "cam": cam,
        "tag": attempt["tag"],
        "stem": stem,
        "method": method,
        "n_frames_loaded": len(frames),
        "correct_distortion": attempt["correct_distortion"],
        "detect_w": DETECT_W,
        "scale_to_native": scale,
        "pass": False,
    }
    if len(frames) < 3:
        result["error"] = "too_few_frames"
        return result
    print(
        f"  frames={len(frames)} detect_w={DETECT_W} scale={scale:.3f} method={method} …",
        flush=True,
    )

    if method == "fov_center":
        out = estimate_H_center_averaged(frames, cam)
        result["estimate_ok"] = bool(out and out.get("H") is not None)
        result["center_circle_detected"] = bool(out and (out.get("types") or {}).get("center_circle"))
        result["y_axis_scale"] = 1.0
        if not out or out.get("H") is None:
            result["error"] = "fov_center_none"
            result["landmark_debug"] = {"ok": False, "error": "fov_center_none"}
            return result
        H_det = out["H"]
        result["landmark_debug"] = {
            "ok": True,
            "note": "fov_filtered_landmarks",
            "cam_fov": cam,
            "frames_ok": out.get("frames_ok"),
            "types": out.get("types"),
            "inliers": out.get("inliers"),
        }
    else:
        est = HomographyEstimator()
        ok = est.estimate_averaged(
            frames,
            correct_distortion=attempt["correct_distortion"],
            min_frames=min(5, len(frames)),
        )
        H_det = est.homography
        result["estimate_ok"] = bool(ok and H_det is not None)
        result["center_circle_detected"] = bool(est.center_circle_detected)
        result["y_axis_scale"] = float(est.y_axis_scale)
        if H_det is None:
            result["error"] = "homography_none"
            result["landmark_debug"] = {"ok": False, "error": "homography_none"}
            return result
        H_det = np.asarray(H_det, dtype=float)
        result["landmark_debug"] = {
            "ok": True,
            "note": "averaged_detect",
            "center_circle_detected": bool(est.center_circle_detected),
            "y_axis_scale": float(est.y_axis_scale),
        }

    H_det = np.asarray(H_det, dtype=float)
    H = scale_H_to_native(H_det, scale)
    h, w = frames[0].shape[:2]
    gate = overlay_gate(H_det, (w, h))
    result["gate"] = gate
    result["pass"] = bool(gate["pass"])
    result["H"] = H.tolist()
    result["native_wh"] = list(native_wh) if native_wh else None
    return result, frames[0], H_det, gate


def calibrate_cam(cam: str, out: Path) -> dict:
    attempts_out = []
    best = None
    for attempt in ATTEMPTS[:3]:
        print(f"{cam}: attempt {attempt['tag']} …", flush=True)
        raw = run_attempt(cam, attempt)
        if isinstance(raw, tuple):
            result, frame0, H, gate = raw
        else:
            result = raw
            frame0 = H = gate = None
        attempts_out.append({k: v for k, v in result.items() if k != "H"})
        if frame0 is not None and H is not None and gate is not None:
            ov = draw_overlay(frame0, H, gate)
            ov_path = out / f"{cam}_{attempt['tag']}_overlay_v2.jpg"
            cv2.imwrite(str(ov_path), ov)
            result["overlay"] = str(ov_path.relative_to(ROOT))
            attempts_out[-1]["overlay"] = result["overlay"]
        if result.get("pass") and result.get("H") is not None:
            best = result
            print(f"{cam}: PASS on {attempt['tag']}", flush=True)
            break
        print(f"{cam}: FAIL {attempt['tag']} reason={result.get('gate', {}).get('reason') or result.get('error')}", flush=True)

    summary = {
        "cam": cam,
        "pass": bool(best and best.get("pass")),
        "attempts": attempts_out,
        "chosen_tag": None if best is None else best.get("tag"),
    }
    if best and best.get("pass"):
        payload = {
            "camera": cam,
            "version": "auto_v2_fov" if best.get("method") == "fov_center" else "auto_v2",
            "pass": True,
            "tag": best["tag"],
            "method": best.get("method"),
            "stem": best["stem"],
            "H": best["H"],
            "source": f"{best.get('method', 'averaged')}+overlay_gate",
            "n_frames": best["n_frames_loaded"],
            "pitch_length_m": PL,
            "pitch_width_m": PW,
            "center_circle_detected": best.get("center_circle_detected"),
            "y_axis_scale": best.get("y_axis_scale"),
            "gate": best.get("gate"),
            "landmark_debug": best.get("landmark_debug"),
            "overlay": best.get("overlay"),
        }
        path = out / f"{cam}_auto_v2.json"
        path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        summary["path"] = str(path.relative_to(ROOT))
        summary["gate"] = best.get("gate")
    else:
        summary["handoff"] = "manual_calib"
        fail_note = out / f"{cam}_auto_v2_FAILED.json"
        fail_note.write_text(json.dumps(summary, indent=2), encoding="utf-8")
        summary["fail_path"] = str(fail_note.relative_to(ROOT))
    return summary


def main() -> int:
    args = parse_args()
    out = args.out if args.out.is_absolute() else (ROOT / args.out)
    out.mkdir(parents=True, exist_ok=True)
    report = {"cams": {}, "all_pass": False}
    for cam in args.cams:
        report["cams"][cam] = calibrate_cam(cam, out)
    report["all_pass"] = all(report["cams"][c].get("pass") for c in args.cams)
    report["next"] = (
        "in_pitch_bounds_and_wire"
        if report["all_pass"]
        else "manual_fallback_handoff"
    )
    path = out / "auto_v2_status.json"
    path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2), flush=True)
    print(f"wrote {path}", flush=True)
    return 0 if report["all_pass"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
