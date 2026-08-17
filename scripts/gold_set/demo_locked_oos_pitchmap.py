#!/usr/bin/env python3
"""OOS dual-pane demo: locked cam video + 2D pitch with ball (x,y).

Uses locked multicam picks from the 4quad det cache. Pitch coords are
FOV-approximate (image→cam FOV wedge) until a real per-cam homography
JSON is present under reports/eval_match2_v10/match2_pitch_calib/.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts" / "gold_set"))

from eval_match2_top_left_multicam_baseline import cache_load  # noqa: E402
from multicam_select_policy import (  # noqa: E402
    PRODUCT_POLICY_ID,
    SURVEY_CAMS,
    TOP_LEFT_THR_BY_CAM,
    StickyCamPicker,
    filter_active,
    pick_product,
)

DETECT_W = 1920
PITCH_LEN_M = 105.0
PITCH_WID_M = 68.0
# SVG pitch space used by camera_pitch_coverage (inner field 880×560)
SVG_W, SVG_H = 880.0, 560.0

# Screenshot-validated FOV wedges (same as camera_pitch_coverage/index.html)
FOV_POLY_SVG = {
    "P1": [[60, 60], [640, 40], [660, 520], [80, 540]],
    "P6": [[100, 40], [820, 20], [860, 360], [120, 540]],
    "P7": [[0, 5], [540, 25], [520, 560], [90, 490]],
    "P8": [[360, 40], [880, 20], [880, 540], [400, 540]],
    "P10": [[40, 0], [520, 0], [560, 360], [40, 300]],
    "P12": [[0, 30], [500, 60], [420, 540], [0, 500]],
    "Cam4plus": [[0, 0], [480, 0], [560, 180], [0, 560]],
    "Cam5plus": [[380, 20], [880, 0], [880, 560], [420, 540]],
}

SLOTS = {
    "bottom_right": {
        "stem": "quad_bottom_right_t00412.0s",
        "label": "Bottom Right OOS — video + pitch (x,y)",
        "clock": "6:52–6:58",
    },
    "top_right": {
        "stem": "quad_top_right_t00125.0s",
        "label": "Top Right OOS — video + pitch (x,y)",
        "clock": "2:05–2:10",
    },
    "center_start": {
        "stem": "quad_center_start_t00008.0s",
        "label": "Center Start OOS — video + pitch (x,y)",
        "clock": "0:08–0:13",
    },
}

SRC = ROOT / "reports/eval_match2_v10/4quad_test/source"
CALIB_DIR = ROOT / "reports/eval_match2_v10/match2_pitch_calib"


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--slot", choices=sorted(SLOTS), default=None)
    p.add_argument("--stem", type=str, default=None)
    p.add_argument("--label", type=str, default=None)
    p.add_argument("--clock", type=str, default=None)
    p.add_argument("--source-dir", type=Path, default=None)
    p.add_argument("--cache", type=Path, default=None)
    p.add_argument("--stride", type=int, default=2)
    p.add_argument("--max-frames", type=int, default=90)
    p.add_argument("--out", type=Path, default=None)
    return p.parse_args()


def resolve_clip(args):
    if args.stem:
        stem = args.stem
        label = args.label or stem
        clock = args.clock or "?"
        source_dir = args.source_dir or SRC
        cache = args.cache
        if cache is None:
            raise SystemExit("--cache required with --stem")
        out = args.out or (ROOT / f"reports/eval_match2_v10/locked_oos_demo_{stem}_pitchmap")
        return stem, label, clock, source_dir, cache, out
    slot = args.slot or "bottom_right"
    meta = SLOTS[slot]
    stem = meta["stem"]
    label = args.label or meta["label"]
    clock = args.clock or meta["clock"]
    source_dir = args.source_dir or SRC
    cache = args.cache or (
        ROOT / f"reports/eval_match2_v10/4quad_multicam_survey/det_cache_{slot}_thr010.json"
    )
    out = args.out or (ROOT / f"reports/eval_match2_v10/locked_oos_demo_{slot}_pitchmap")
    return stem, label, clock, source_dir, cache, out


def resize_w(frame, width=DETECT_W):
    h, w = frame.shape[:2]
    if w == width:
        return frame
    return cv2.resize(frame, (width, int(round(h * width / w))), interpolation=cv2.INTER_AREA)


def svg_to_m(sx, sy):
    # origin top-left of field in SVG → pitch meters with origin at center
    x = (sx / SVG_W) * PITCH_LEN_M - PITCH_LEN_M / 2
    y = (sy / SVG_H) * PITCH_WID_M - PITCH_WID_M / 2
    return float(x), float(y)


def load_homography(cam: str):
    for name in (f"{cam}_manual.json", f"{cam}_top_left_auto.json"):
        path = CALIB_DIR / name
        if not path.is_file():
            continue
        data = json.loads(path.read_text(encoding="utf-8"))
        H = data.get("homography") or data.get("H")
        if H is None:
            continue
        return np.asarray(H, dtype=float), "manual" if "manual" in name else "auto_untrusted"
    return None, "fov_approx"


def fov_H(cam: str, frame_wh):
    """Map full image rectangle → FOV quad in SVG, then compose to meters."""
    w, h = frame_wh
    poly = FOV_POLY_SVG.get(cam)
    if not poly or len(poly) < 4:
        return None
    # order: TL, TR, BR, BL of FOV (approx from poly extremes)
    pts = np.asarray(poly, dtype=float)
    # pick 4 corners by extremes
    tl = pts[np.argmin(pts[:, 0] + pts[:, 1])]
    tr = pts[np.argmin(-pts[:, 0] + pts[:, 1])]
    br = pts[np.argmax(pts[:, 0] + pts[:, 1])]
    bl = pts[np.argmin(pts[:, 0] - pts[:, 1])]
    dst_svg = np.float32([tl, tr, br, bl])
    src = np.float32([[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]])
    H_img_svg = cv2.getPerspectiveTransform(src, dst_svg)
    # SVG → meters (affine)
    # [x_m]   [LEN/SVG_W   0    -LEN/2] [sx]
    # [y_m] = [0   WID/SVG_H    -WID/2] [sy]
    # [1  ]   [0           0         1] [1 ]
    A = np.array(
        [
            [PITCH_LEN_M / SVG_W, 0, -PITCH_LEN_M / 2],
            [0, PITCH_WID_M / SVG_H, -PITCH_WID_M / 2],
            [0, 0, 1],
        ],
        dtype=float,
    )
    return A @ H_img_svg


def pixel_to_pitch(H, cx, cy):
    v = H @ np.array([cx, cy, 1.0], dtype=float)
    if abs(v[2]) < 1e-9:
        return None
    return float(v[0] / v[2]), float(v[1] / v[2])


def draw_pitch(panel_w, panel_h, ball_xy, cam, mode, trail):
    vis = np.zeros((panel_h, panel_w, 3), dtype=np.uint8)
    vis[:] = (32, 48, 36)
    margin = 28
    x0, y0 = margin, margin
    pw, ph = panel_w - 2 * margin, panel_h - 2 * margin
    # grass
    for i in range(10):
        x = x0 + int(i * pw / 10)
        x2 = x0 + int((i + 1) * pw / 10)
        color = (55, 130, 55) if i % 2 == 0 else (48, 118, 48)
        cv2.rectangle(vis, (x, y0), (x2, y0 + ph), color, -1)
    cv2.rectangle(vis, (x0, y0), (x0 + pw, y0 + ph), (240, 240, 240), 2)
    cv2.line(vis, (x0 + pw // 2, y0), (x0 + pw // 2, y0 + ph), (240, 240, 240), 2)
    r = int(9.15 / PITCH_LEN_M * pw)
    cv2.circle(vis, (x0 + pw // 2, y0 + ph // 2), r, (240, 240, 240), 2)
    # boxes
    box_d = int(16.5 / PITCH_LEN_M * pw)
    box_h = int(40.32 / PITCH_WID_M * ph)
    by = y0 + (ph - box_h) // 2
    cv2.rectangle(vis, (x0, by), (x0 + box_d, by + box_h), (240, 240, 240), 2)
    cv2.rectangle(vis, (x0 + pw - box_d, by), (x0 + pw, by + box_h), (240, 240, 240), 2)

    def m_to_px(xm, ym):
        px = x0 + int((xm + PITCH_LEN_M / 2) / PITCH_LEN_M * pw)
        py = y0 + int((ym + PITCH_WID_M / 2) / PITCH_WID_M * ph)
        return px, py

    for i, (xm, ym) in enumerate(trail[-40:]):
        p = m_to_px(xm, ym)
        cv2.circle(vis, p, 3, (80, 180, 255), -1)
    if ball_xy is not None:
        p = m_to_px(*ball_xy)
        cv2.circle(vis, p, 10, (0, 255, 255), -1)
        cv2.circle(vis, p, 12, (0, 0, 0), 2)
        tag = f"pitch  x={ball_xy[0]:+.1f}m  y={ball_xy[1]:+.1f}m"
    else:
        tag = "pitch  no ball"
    cv2.rectangle(vis, (8, 8), (min(panel_w - 8, 620), 78), (0, 0, 0), -1)
    cv2.putText(vis, tag, (16, 36), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 255), 2)
    cv2.putText(
        vis,
        f"{cam or 'none'}  map={mode}",
        (16, 64),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (200, 200, 200),
        2,
    )
    return vis


def paint_video(frame, cam, pred, overlay_w=960):
    vis = resize_w(frame, overlay_w)
    h, w = vis.shape[:2]
    scale = overlay_w / float(frame.shape[1])
    if pred is None:
        cv2.rectangle(vis, (8, 8), (min(w - 8, 520), 56), (0, 0, 0), -1)
        cv2.putText(vis, f"{cam or 'none'}  no ball", (16, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 140, 255), 2)
        return vis
    box, conf, side = pred
    x, y, bw, bh = [int(round(v * scale)) for v in box]
    color = (0, 255, 0) if conf >= 0.80 else (0, 220, 255)
    cv2.rectangle(vis, (x, y), (x + max(bw, 1), y + max(bh, 1)), color, 3)
    cx, cy = x + max(bw, 1) // 2, y + max(bh, 1) // 2
    cv2.circle(vis, (cx, cy), 10, color, -1)
    cv2.rectangle(vis, (8, 8), (min(w - 8, 700), 56), (0, 0, 0), -1)
    cv2.putText(
        vis,
        f"{cam}  conf={conf:.2f}  px=({int(box[0]+box[2]/2)},{int(box[1]+box[3]/2)})",
        (16, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        color,
        2,
    )
    return vis


def encode_h264(path: Path):
    tmp = path.with_name(path.stem + "_h264.mp4")
    cmd = [
        "ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
        "-i", str(path), "-c:v", "libx264", "-pix_fmt", "yuv420p",
        "-preset", "veryfast", "-crf", "20", "-movflags", "+faststart", "-an", str(tmp),
    ]
    subprocess.run(cmd, check=True)
    if not tmp.is_file():
        raise RuntimeError(f"ffmpeg failed to write {tmp}")
    tmp.replace(path)


def main() -> int:
    args = parse_args()
    stem, label, clock, source_dir, cache, out = resolve_clip(args)
    source_dir = source_dir if source_dir.is_absolute() else (ROOT / source_dir)
    cache = cache if cache.is_absolute() else (ROOT / cache)
    out = out if out.is_absolute() else (ROOT / out)
    dets = cache_load(cache)
    n_cache = len(next(iter(dets.values())))
    n = min(args.max_frames * args.stride, n_cache)
    out.mkdir(parents=True, exist_ok=True)

    caps = {}
    for cam in SURVEY_CAMS:
        path = source_dir / f"{stem}_{cam}.mp4"
        cap = cv2.VideoCapture(str(path))
        if not cap.isOpened():
            raise RuntimeError(f"open failed {path}")
        caps[cam] = cap
    fps = float(caps[SURVEY_CAMS[0]].get(cv2.CAP_PROP_FPS) or 30.0) / args.stride

    H_by_cam = {}
    mode_by_cam = {}
    for cam in SURVEY_CAMS:
        H, mode = load_homography(cam)
        if H is None or mode == "auto_untrusted":
            H = None
            mode = "fov_approx"
        H_by_cam[cam] = H
        mode_by_cam[cam] = mode

    ok, fr0 = caps[SURVEY_CAMS[0]].read()
    caps[SURVEY_CAMS[0]].set(cv2.CAP_PROP_POS_FRAMES, 0)
    if ok:
        sample = resize_w(fr0, DETECT_W)
        wh = (sample.shape[1], sample.shape[0])
        for cam in SURVEY_CAMS:
            if H_by_cam[cam] is None:
                H_by_cam[cam] = fov_H(cam, wh)
                mode_by_cam[cam] = "fov_approx"

    writer = None
    raw_path = out / f"{stem}_video_pitch.mp4"
    track = []
    trail = []
    written = 0
    sticky = StickyCamPicker()

    for i in range(n):
        frames = {}
        ok_all = True
        for cam, cap in caps.items():
            ok, fr = cap.read()
            if not ok:
                ok_all = False
                break
            frames[cam] = resize_w(fr, DETECT_W)
        if not ok_all:
            break
        if i % args.stride != 0:
            continue

        active = filter_active(dets, i, SURVEY_CAMS, TOP_LEFT_THR_BY_CAM)
        if active:
            raw_cam, raw_pred = pick_product(active, frames_by_cam=frames)
        else:
            raw_cam, raw_pred = None, None
        cam, pred = sticky.step(raw_cam, raw_pred)
        emit_cam, emit_pred = sticky.emit(cam, pred)

        ball_xy = None
        mode = "none"
        if emit_cam and emit_pred is not None and H_by_cam.get(emit_cam) is not None:
            box, conf, side = emit_pred
            cx = box[0] + box[2] / 2
            cy = box[1] + box[3] / 2
            ball_xy = pixel_to_pitch(H_by_cam[emit_cam], cx, cy)
            mode = mode_by_cam.get(emit_cam, "?")
            if ball_xy is not None:
                trail.append(ball_xy)

        paint_cam = cam if cam in frames else next(iter(frames))
        left = paint_video(frames[paint_cam], cam, pred, 960)
        right = draw_pitch(960, left.shape[0], ball_xy, emit_cam, mode, trail)
        combo = np.hstack([left, right])
        if writer is None:
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            writer = cv2.VideoWriter(str(raw_path), fourcc, fps, (combo.shape[1], combo.shape[0]))
        writer.write(combo)
        track.append(
            {
                "i": i,
                "cam": cam,
                "conf": None if pred is None else float(pred[1]),
                "px": None
                if pred is None
                else [float(pred[0][0] + pred[0][2] / 2), float(pred[0][1] + pred[0][3] / 2)],
                "pitch_m": None if ball_xy is None else [round(ball_xy[0], 2), round(ball_xy[1], 2)],
                "map": mode,
            }
        )
        written += 1
        if written >= args.max_frames:
            break

    for cap in caps.values():
        cap.release()
    if writer is not None:
        writer.release()
        encode_h264(raw_path)

    (out / "track.json").write_text(json.dumps(track, indent=2), encoding="utf-8")
    stats = {
        "stem": stem,
        "label": label,
        "clock": clock,
        "policy": PRODUCT_POLICY_ID,
        "n_frames": written,
        "map_note": (
            "Pitch (x,y) uses FOV-approximate projection from camera_pitch_coverage wedges "
            "until Cam*_manual.json homographies exist. Absolute meters are provisional."
        ),
        "overlay": str(raw_path.relative_to(ROOT)),
    }
    (out / "stats.json").write_text(json.dumps(stats, indent=2), encoding="utf-8")
    html = f"""<!DOCTYPE html>
<html><head><meta charset="utf-8"/><title>{label}</title>
<style>
body{{margin:0;font-family:IBM Plex Sans,Segoe UI,sans-serif;background:#121612;color:#eef2ee}}
main{{max-width:1200px;margin:0 auto;padding:1.2rem}}
h1{{font-size:1.25rem;margin:0 0 .4rem}}
.sub{{color:#9aab9a;font-size:.9rem;line-height:1.4;margin-bottom:1rem}}
video{{width:100%;background:#000;border-radius:8px}}
code{{color:#e8c547}}
</style></head><body><main>
<h1>{label}</h1>
<p class="sub">{clock} · locked pick <code>{PRODUCT_POLICY_ID}</code><br/>
Left: selected cam + ball box. Right: 2D pitch with ball <b>x,y meters</b> (provisional FOV map until manual H).<br/>
{stats['map_note']}</p>
<video controls autoplay muted loop src="/{raw_path.relative_to(ROOT).as_posix()}"></video>
<p class="sub">Track JSON: <code>/{(out/'track.json').relative_to(ROOT).as_posix()}</code></p>
</main></body></html>
"""
    (out / "index.html").write_text(html, encoding="utf-8")
    (out / "readme.md").write_text(
        f"# {label}\n\nOpen: http://127.0.0.1:8080/{out.relative_to(ROOT).as_posix()}/\n",
        encoding="utf-8",
    )
    print(json.dumps(stats, indent=2), flush=True)
    print(f"wrote {out / 'index.html'}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
