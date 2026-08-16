#!/usr/bin/env python3
"""5x5 clips: 5 random 5-second Match 2 windows, product ball stack.

Detect 0.3 → ByteTrack → emit 0.80. Optional best-camera pick across
all Match 2 views. Writes source clips, overlays, JSON, HTML.
Never trains. Real Match 2 footage only.
"""
from __future__ import annotations

import argparse
import json
import random
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts" / "gold_set"))

from eval_match2_v10_video_system import (  # noqa: E402
    DEFAULT_CKPT,
    MASTER_CAMS,
    load_ball_model,
    make_prelabeler,
    make_tracker,
    pick_selected,
    raw_emit_from_dets,
)

CLIP_SEC = 5.0
N_CLIPS = 5
PACK_NAME = "5x5_clips"
QUAD_PACK = "4quad_test"
MOSAIC_CELL_W = 480
QUAD_CLIPS = [
    {"label": "Center Start", "start_clock": "0:08", "end_clock": "0:13", "slot": "center_start"},
    {"label": "Bottom Right", "start_clock": "6:52", "end_clock": "6:58", "slot": "bottom_right"},
    {"label": "Top Left", "start_clock": "0:26", "end_clock": "0:31", "slot": "top_left"},
    {"label": "Top Right", "start_clock": "2:05", "end_clock": "2:10", "slot": "top_right"},
]
QUAD_HTML_ORDER = ["top_left", "top_right", "center_start", "bottom_right"]
MATCH2_CAMS = [
    ("P1", ROOT / "data/raw/Match 2/Cam 3-P1.mp4"),
    ("P6", ROOT / "data/raw/Match 2/Cam 6-P6-002.mp4"),
    ("P7", ROOT / "data/raw/Match 2/Cam 11-P7-003.mp4"),
    ("P8", ROOT / "data/raw/Match 2/Cam 14-P8-001.mp4"),
    ("P10", ROOT / "data/raw/Match 2/Cam 8-P10-003.mp4"),
    ("P12", ROOT / "data/raw/Match 2/Cam 10-P12-001.mp4"),
    ("Cam4plus", ROOT / "data/raw/Match 2/Cam 4+-002.mp4"),
    ("Cam5plus", ROOT / "data/raw/Match 2/Cam 5+-004.mp4"),
]


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--ball-checkpoint", type=Path, default=DEFAULT_CKPT)
    p.add_argument("--n-clips", type=int, default=N_CLIPS)
    p.add_argument("--clip-sec", type=float, default=CLIP_SEC)
    p.add_argument("--seed", type=int, default=20260813)
    p.add_argument("--stride", type=int, default=2)
    p.add_argument("--min-thr", type=float, default=0.30)
    p.add_argument("--emit-thresh", type=float, default=0.80)
    p.add_argument("--track-thresh", type=float, default=0.10)
    p.add_argument("--ema-alpha", type=float, default=0.3)
    p.add_argument("--overlay-width", type=int, default=1280)
    p.add_argument(
        "--select-camera",
        choices=["random", "max_conf", "size_weighted", "largest_ball"],
        default="random",
    )
    p.add_argument(
        "--cameras",
        choices=["masters", "all_match2"],
        default="masters",
    )
    p.add_argument("--quad-test", action="store_true", help="4 named Match 2 windows, 2x2 dashboard")
    p.add_argument("--skip-extract", action="store_true", help="reuse source clips if present")
    p.add_argument(
        "--out",
        type=Path,
        default=ROOT / "reports" / "eval_match2_v10" / PACK_NAME,
    )
    return p.parse_args()


def parse_clock(text: str) -> float:
    parts = str(text).split(":")
    if len(parts) != 2:
        raise ValueError(f"expected m:ss, got {text}")
    return int(parts[0]) * 60 + float(parts[1])


def fmt_clock(sec: float) -> str:
    total = int(round(float(sec)))
    return f"{total // 60}:{total % 60:02d}"


def slug_label(label: str) -> str:
    return "".join(ch.lower() if ch.isalnum() else "_" for ch in label).strip("_")


def camera_roster(kind: str):
    if kind == "all_match2":
        return list(MATCH2_CAMS)
    return list(MASTER_CAMS)


def video_duration_sec(path: Path) -> float:
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise FileNotFoundError(f"cannot open {path}")
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 30.0)
    n = float(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0.0)
    cap.release()
    if fps <= 0 or n <= 0:
        raise RuntimeError(f"bad duration for {path}")
    return n / fps


def windows_overlap(cam: str, start: float, dur: float, taken: list) -> bool:
    for row in taken:
        if row.get("camera") != cam:
            continue
        if abs(row["start_sec"] - start) < dur:
            return True
    return False


def starts_overlap(start: float, dur: float, taken: list) -> bool:
    for row in taken:
        if abs(row["start_sec"] - start) < dur:
            return True
    return False


def pick_random_windows(n: int, clip_sec: float, seed: int) -> list[dict]:
    rng = random.Random(seed)
    durations = {name: video_duration_sec(path) for name, path in MASTER_CAMS}
    taken: list[dict] = []
    for _ in range(2000):
        if len(taken) >= n:
            return taken
        name, path = MASTER_CAMS[rng.randrange(len(MASTER_CAMS))]
        length = durations[name]
        if length <= clip_sec + 1.0:
            continue
        start = rng.uniform(0.0, length - clip_sec)
        if windows_overlap(name, start, clip_sec, taken):
            continue
        taken.append(
            {
                "camera": name,
                "video_path": str(path),
                "start_sec": round(start, 3),
                "duration_sec": clip_sec,
            }
        )
    raise RuntimeError(f"could not pick {n} non-overlapping {clip_sec}s windows")


def pick_synced_windows(spec: dict, cams: list) -> list[dict]:
    rng = random.Random(spec["seed"])
    shortest = min(video_duration_sec(path) for _name, path in cams)
    clip_sec = spec["clip_sec"]
    if shortest <= clip_sec + 1.0:
        raise RuntimeError("camera files shorter than clip length")
    taken: list[dict] = []
    for _ in range(2000):
        if len(taken) >= spec["n"]:
            return taken
        start = rng.uniform(0.0, shortest - clip_sec)
        if starts_overlap(start, clip_sec, taken):
            continue
        taken.append(synced_window(round(start, 3), clip_sec, cams))
    raise RuntimeError(f"could not pick {spec['n']} synced {clip_sec}s windows")


def synced_window(start: float, clip_sec: float, cams: list) -> dict:
    return {
        "camera": "bestcam",
        "start_sec": start,
        "duration_sec": clip_sec,
        "cameras": [
            {"camera": name, "video_path": str(path)} for name, path in cams
        ],
    }


def clip_stem(index: int, window: dict) -> str:
    start = window["start_sec"]
    if window.get("label"):
        return f"quad_{slug_label(window['label'])}_t{start:07.1f}s"
    cam = window.get("camera", "bestcam")
    return f"5x5_clip_{index:02d}_{cam}_t{start:07.1f}s"


def named_quad_windows(cams: list) -> list[dict]:
    shortest = min(video_duration_sec(path) for _name, path in cams)
    windows = []
    for spec in QUAD_CLIPS:
        start = parse_clock(spec["start_clock"])
        end = parse_clock(spec["end_clock"])
        if end <= start:
            raise ValueError(f"bad window {spec}")
        if end > shortest:
            raise RuntimeError(f"{spec['label']} ends at {end}s but shortest cam is {shortest:.1f}s")
        win = synced_window(start, round(end - start, 3), cams)
        win["label"] = spec["label"]
        win["slot"] = spec["slot"]
        windows.append(win)
    return windows


def cam_window(synced: dict, cam_row: dict) -> dict:
    return {
        "camera": cam_row["camera"],
        "video_path": cam_row["video_path"],
        "start_sec": synced["start_sec"],
        "duration_sec": synced["duration_sec"],
    }


def extract_clip(window: dict, dest: Path, skip_extract: bool = False) -> Path:
    if skip_extract and dest.is_file() and dest.stat().st_size > 1000:
        return dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        "ffmpeg",
        "-y",
        "-hide_banner",
        "-loglevel",
        "error",
        "-ss",
        f"{window['start_sec']:.3f}",
        "-i",
        window["video_path"],
        "-t",
        f"{window['duration_sec']:.3f}",
        "-c:v",
        "libx264",
        "-crf",
        "18",
        "-preset",
        "veryfast",
        "-an",
        str(dest),
    ]
    subprocess.run(cmd, check=True)
    if not dest.is_file():
        raise RuntimeError(f"ffmpeg did not write {dest}")
    return dest


def resize_overlay(frame, width: int):
    if width <= 0:
        return frame
    h, w = frame.shape[:2]
    if w <= width:
        return frame
    scale = width / float(w)
    return cv2.resize(frame, (width, int(round(h * scale))))


def scale_pred(pred, sx: float, sy: float):
    if pred is None:
        return None
    box, conf, side = pred
    x, y, w, h = box
    return ([x * sx, y * sy, w * sx, h * sy], conf, side * min(sx, sy))


def paint_view(frame, raw_pred, emit_pred, label: str, width: int):
    vis = resize_overlay(frame, width)
    sx = vis.shape[1] / float(frame.shape[1])
    sy = vis.shape[0] / float(frame.shape[0])
    return draw_raw_and_emit(
        vis,
        scale_pred(raw_pred, sx, sy),
        scale_pred(emit_pred, sx, sy),
        label,
    )


def draw_raw_and_emit(frame, raw_pred, emit_pred, label: str):
    vis = frame.copy()
    emitted = emit_pred is not None
    pred = emit_pred if emitted else raw_pred
    color = (0, 255, 0) if emitted else (0, 140, 255)
    h, w = vis.shape[:2]
    thick = max(4, int(round(0.008 * min(h, w))))
    font = max(0.5, min(0.85, h / 800.0))
    if pred is None:
        return vis
    _box, conf, side = pred
    x, y, bw, bh = [int(round(v)) for v in pred[0]]
    cv2.rectangle(vis, (x, y), (x + max(bw, 1), y + max(bh, 1)), color, thick)
    cx, cy = x + max(bw, 1) // 2, y + max(bh, 1) // 2
    radius = max(16, int(max(bw, bh) * 1.6), int(0.03 * min(h, w)))
    cv2.circle(vis, (cx, cy), radius, color, thick)
    cv2.drawMarker(vis, (cx, cy), color, cv2.MARKER_CROSS, radius, thick)
    kind = "EMIT" if emitted else "RAW"
    tag = f"{kind} {conf:.2f}"
    tx = min(w - 8, cx + radius + 8)
    ty = max(24, cy - radius)
    cv2.putText(vis, tag, (tx, ty), cv2.FONT_HERSHEY_SIMPLEX, font, color, 2)
    return vis


def pred_list(pred):
    return [pred] if pred is not None else []


def new_track_state(path: Path, args, overlay_path: Path | None, label: str | None = None) -> dict:
    return {
        "pre": make_prelabeler(load_track_model(args), args.min_thr, True),
        "tracker": make_tracker(args),
        "writer": None,
        "overlay_path": overlay_path,
        "label": label or path.stem,
        "raws": [],
        "emits": [],
        "n_frames": 0,
        "n_emit": 0,
        "n_raw": 0,
        "confs": [],
        "last_emit": None,
        "last_raw": None,
        "args": args,
    }


def step_track(state: dict, frame):
    state["n_frames"] += 1
    if (state["n_frames"] - 1) % max(1, state["args"].stride) == 0:
        row = raw_emit_from_dets(state["pre"], state["tracker"], frame)
        state["last_emit"] = row["emit"][0] if row["emit"] else None
        state["last_raw"] = row["raw"][0] if row["raw"] else None
        if row["raw"]:
            state["n_raw"] += 1
    if state["last_emit"] is not None:
        state["n_emit"] += 1
        state["confs"].append(float(state["last_emit"][1]))
    state["raws"].append(state["last_raw"])
    state["emits"].append(state["last_emit"])
    maybe_write_overlay(state, frame)
    top = state["last_emit"][1] if state["last_emit"] is not None else (
        state["last_raw"][1] if state["last_raw"] else 0.0
    )
    print(
        f"  {state['label']} f={state['n_frames']:03d} emit={state['last_emit'] is not None} "
        f"raw={state['last_raw'] is not None} top={top:.3f}",
        flush=True,
    )


def track_frames(path: Path, args, overlay_path: Path | None, label: str | None = None) -> dict:
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise RuntimeError(f"open failed: {path}")
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 30.0)
    state = new_track_state(path, args, overlay_path, label=label)
    state["fps"] = fps
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        step_track(state, frame)
    cap.release()
    if state["writer"] is not None:
        state["writer"].release()
        encode_browser_mp4(overlay_path)
    return finish_track(state, path, fps)


def maybe_write_overlay(state: dict, frame):
    path = state["overlay_path"]
    if path is None:
        return
    vis = paint_view(
        frame,
        state["last_raw"],
        state["last_emit"],
        state["label"],
        state["args"].overlay_width,
    )
    if state["writer"] is None:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        state["writer"] = cv2.VideoWriter(
            str(path), fourcc, state.get("fps") or 30.0, (vis.shape[1], vis.shape[0])
        )
    state["writer"].write(vis)


def finish_track(state: dict, path: Path, fps: float) -> dict:
    n_frames = state["n_frames"]
    confs = state["confs"]
    return {
        "n_frames": n_frames,
        "n_emit_held": state["n_emit"],
        "n_raw_hits": state["n_raw"],
        "emit_rate": (state["n_emit"] / n_frames) if n_frames else 0.0,
        "mean_emit_conf": (sum(confs) / len(confs)) if confs else None,
        "overlay": str(state["overlay_path"]) if state["overlay_path"] else None,
        "source": str(path),
        "fps": fps,
        "stride": state["args"].stride,
        "raws": state["raws"],
        "emits": state["emits"],
    }


def track_clip(path: Path, args, overlay_path: Path) -> dict:
    result = track_frames(path, args, overlay_path)
    return {k: result[k] for k in result if k not in ("raws", "emits")}


_MODEL = None


def load_track_model(args):
    global _MODEL
    if _MODEL is None:
        ckpt = args.ball_checkpoint
        if not ckpt.is_file():
            raise FileNotFoundError(f"missing checkpoint: {ckpt}")
        print(f"loading {ckpt}", flush=True)
        _MODEL = load_ball_model(str(ckpt))
    return _MODEL


def open_caps(paths: list):
    caps = [cv2.VideoCapture(str(path)) for path in paths]
    for cap, path in zip(caps, paths):
        if not cap.isOpened():
            raise RuntimeError(f"open failed: {path}")
    return caps


def close_caps(caps: list):
    for cap in caps:
        cap.release()


def read_synced_frames(caps: list):
    frames = []
    for cap in caps:
        ok, frame = cap.read()
        if not ok:
            return None
        frames.append(frame)
    return frames


def add_winner_border(cell, is_winner: bool):
    if not is_winner:
        return cell
    out = cell.copy()
    cv2.rectangle(out, (2, 2), (out.shape[1] - 3, out.shape[0] - 3), (0, 255, 0), 6)
    return out


def tile_mosaic(cells: list):
    cols = 4
    h, w = cells[0].shape[:2]
    rows = (len(cells) + cols - 1) // cols
    sheet = np.zeros((rows * h, cols * w, 3), dtype=np.uint8)
    for i, cell in enumerate(cells):
        r, c = divmod(i, cols)
        ch, cw = cell.shape[:2]
        sheet[r * h : r * h + ch, c * w : c * w + cw] = cell
    return sheet


def empty_win_counts(names: list) -> dict:
    counts = {name: 0 for name in names}
    counts["none"] = 0
    return counts


def compose_bestcam(job: dict, args) -> dict:
    names = job["names"]
    tracks = job["tracks"]
    n_frames = min(tracks[name]["n_frames"] for name in names)
    fps = tracks[names[0]]["fps"]
    caps = open_caps(job["paths"])
    acc = {
        "sel_writer": None,
        "mosaic_writer": None,
        "wins": empty_win_counts(names),
        "n_emit": 0,
        "n_raw": 0,
        "confs": [],
        "n_frames": 0,
        "fps": fps,
    }
    for idx in range(n_frames):
        frames = read_synced_frames(caps)
        if frames is None:
            break
        step_bestcam(acc, {"frames": frames, "idx": idx, "job": job, "args": args})
    close_caps(caps)
    if acc["sel_writer"] is not None:
        acc["sel_writer"].release()
        encode_browser_mp4(job["out_sel"])
    if acc["mosaic_writer"] is not None:
        acc["mosaic_writer"].release()
        encode_browser_mp4(job["out_mosaic"])
    return bestcam_stats(acc, job)


def step_bestcam(acc: dict, step: dict):
    job = step["job"]
    args = step["args"]
    idx = step["idx"]
    picked = select_frame(job["tracks"], {"names": job["names"], "idx": idx, "mode": args.select_camera})
    acc["n_frames"] += 1
    acc["wins"][picked["cam"] if picked["cam"] else "none"] += 1
    if picked["raw"] is not None:
        acc["n_raw"] += 1
        acc["confs"].append(float(picked["raw"][1]))
    if picked["emit"] is not None:
        acc["n_emit"] += 1
    vis = selected_view({"frames": step["frames"], "names": job["names"], "picked": picked, "args": args})
    mosaic = mosaic_view({"frames": step["frames"], "names": job["names"], "winner": picked["cam"], "tracks": job["tracks"], "idx": idx})
    acc["sel_writer"] = ensure_writer(acc["sel_writer"], {"path": job["out_sel"], "fps": acc["fps"], "frame": vis})
    acc["mosaic_writer"] = ensure_writer(acc["mosaic_writer"], {"path": job["out_mosaic"], "fps": acc["fps"], "frame": mosaic})
    acc["sel_writer"].write(vis)
    acc["mosaic_writer"].write(mosaic)
    top = picked["raw"][1] if picked["raw"] is not None else 0.0
    print(
        f"  bestcam f={idx + 1:03d} cam={picked['cam'] or 'none'} "
        f"emit={picked['emit'] is not None} raw={picked['raw'] is not None} top={top:.3f}",
        flush=True,
    )


def select_frame(tracks: dict, spec: dict):
    names = spec["names"]
    idx = spec["idx"]
    pred_map = {name: pred_list(tracks[name]["raws"][idx]) for name in names}
    cam, raw_pred = pick_selected(pred_map, spec["mode"])
    emit_pred = tracks[cam]["emits"][idx] if cam else None
    return {"cam": cam, "raw": raw_pred, "emit": emit_pred}


def selected_view(job: dict):
    frames = job["frames"]
    picked = job["picked"]
    args = job["args"]
    if picked["cam"] is None:
        return paint_view(frames[0], None, None, "none", args.overlay_width)
    frame = frames[job["names"].index(picked["cam"])]
    return paint_view(frame, picked["raw"], picked["emit"], picked["cam"], args.overlay_width)


def mosaic_view(job: dict):
    cells = []
    for name, frame in zip(job["names"], job["frames"]):
        raw_pred = job["tracks"][name]["raws"][job["idx"]]
        emit_pred = job["tracks"][name]["emits"][job["idx"]]
        vis = paint_view(frame, raw_pred, emit_pred, name, MOSAIC_CELL_W)
        cells.append(add_winner_border(vis, name == job["winner"]))
    return tile_mosaic(cells)


def encode_browser_mp4(path: Path) -> Path:
    tmp = path.with_name(path.stem + "_h264.mp4")
    cmd = [
        "ffmpeg",
        "-y",
        "-hide_banner",
        "-loglevel",
        "error",
        "-i",
        str(path),
        "-c:v",
        "libx264",
        "-pix_fmt",
        "yuv420p",
        "-preset",
        "veryfast",
        "-crf",
        "20",
        "-movflags",
        "+faststart",
        "-an",
        str(tmp),
    ]
    subprocess.run(cmd, check=True)
    tmp.replace(path)
    return path


def ensure_writer(writer, spec: dict):
    if writer is not None:
        return writer
    spec["path"].parent.mkdir(parents=True, exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    frame = spec["frame"]
    return cv2.VideoWriter(str(spec["path"]), fourcc, spec["fps"], (frame.shape[1], frame.shape[0]))


def bestcam_stats(acc: dict, job: dict):
    wins = acc["wins"]
    n_frames = acc["n_frames"]
    confs = acc["confs"]
    top_cam = max((k for k in wins if k != "none"), key=lambda k: wins[k], default="none")
    return {
        "n_frames": n_frames,
        "n_emit_held": acc["n_emit"],
        "n_raw_hits": acc["n_raw"],
        "emit_rate": (acc["n_emit"] / n_frames) if n_frames else 0.0,
        "mean_selected_conf": (sum(confs) / len(confs)) if confs else None,
        "win_counts": wins,
        "top_camera": top_cam,
        "overlay": str(job["out_sel"]),
        "mosaic": str(job["out_mosaic"]),
        "per_camera": {
            name: {
                "n_raw_hits": job["tracks"][name]["n_raw_hits"],
                "n_emit_held": job["tracks"][name]["n_emit_held"],
                "emit_rate": job["tracks"][name]["emit_rate"],
                "mean_emit_conf": job["tracks"][name]["mean_emit_conf"],
                "n_frames": job["tracks"][name]["n_frames"],
            }
            for name in job["names"]
        },
    }


def extract_synced_sources(window: dict, src_dir: Path, stem: str, skip_extract: bool = False) -> dict:
    paths = {}
    for cam_row in window["cameras"]:
        name = cam_row["camera"]
        dest = src_dir / f"{stem}_{name}.mp4"
        extract_clip(cam_window(window, cam_row), dest, skip_extract=skip_extract)
        paths[name] = dest
    return paths


def track_synced_cameras(paths: dict, args, ov_dir: Path, stem: str) -> dict:
    tracks = {}
    for name, src in paths.items():
        overlay = ov_dir / f"{stem}_{name}_boxes.mp4"
        print(f"{src.stem}: tracking {src}", flush=True)
        tracks[name] = track_frames(src, args, overlay, label=name)
    return tracks


def run_random(args, out: Path) -> dict:
    src_dir = out / "source"
    ov_dir = out / "overlay"
    src_dir.mkdir(parents=True, exist_ok=True)
    ov_dir.mkdir(parents=True, exist_ok=True)
    windows = pick_random_windows(args.n_clips, args.clip_sec, args.seed)
    clips = []
    for i, window in enumerate(windows, start=1):
        stem = clip_stem(i, window)
        src = extract_clip(window, src_dir / f"{stem}.mp4")
        print(f"{stem}: tracking {src}", flush=True)
        stats = track_clip(src, args, ov_dir / f"{stem}_overlay.mp4")
        clips.append({**window, "stem": stem, "stats": stats})
    return {"clips": clips}


def run_bestcam(args, out: Path) -> dict:
    src_dir = out / "source"
    ov_dir = out / "overlay"
    src_dir.mkdir(parents=True, exist_ok=True)
    ov_dir.mkdir(parents=True, exist_ok=True)
    cams = camera_roster(args.cameras)
    spec = {"n": args.n_clips, "clip_sec": args.clip_sec, "seed": args.seed}
    if args.quad_test:
        windows = named_quad_windows(cams)
    else:
        windows = pick_synced_windows(spec, cams)
    clips = []
    for i, window in enumerate(windows, start=1):
        stem = clip_stem(i, window)
        paths = extract_synced_sources(window, src_dir, stem, skip_extract=args.skip_extract)
        tracks = track_synced_cameras(paths, args, ov_dir, stem)
        names = [row["camera"] for row in window["cameras"]]
        job = {
            "names": names,
            "paths": [paths[name] for name in names],
            "tracks": tracks,
            "out_sel": ov_dir / f"{stem}_overlay.mp4",
            "out_mosaic": ov_dir / f"{stem.replace('bestcam', 'mosaic')}.mp4",
        }
        stats = compose_bestcam(job, args)
        clips.append({**window, "stem": stem, "stats": stats})
    return {"clips": clips}


def fmt_mean(value):
    return f"{value:.3f}" if value is not None else "—"


def write_summary(out_dir: Path, payload: dict) -> Path:
    path = out_dir / "5x5_clips_summary.json"
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    md = out_dir / "5x5_clips_summary.md"
    md.write_text(summary_markdown(payload), encoding="utf-8")
    if payload.get("select_camera") != "random":
        write_html(out_dir, payload)
    return path


def summary_markdown(payload: dict) -> str:
    lines = [
        f"# {payload.get('title', '5x5 ball clips')}",
        "",
        f"Seed `{payload['seed']}` · {payload['n_clips']} clips · "
        f"detect {payload['min_thr']} → emit {payload['emit_thresh']}",
        f"Camera set `{payload.get('cameras', 'masters')}` · "
        f"select `{payload.get('select_camera', 'random')}`",
        "",
    ]
    if payload.get("select_camera") == "random":
        lines.extend(random_table(payload))
    else:
        lines.extend(bestcam_table(payload))
    return "\n".join(lines) + "\n"


def random_table(payload: dict) -> list:
    lines = [
        "| Clip | Camera | Start | Raw hit frames | Emit hold rate | Mean emit conf |",
        "|------|--------|-------|----------------|----------------|----------------|",
    ]
    for clip in payload["clips"]:
        mean_txt = fmt_mean(clip["stats"].get("mean_emit_conf"))
        lines.append(
            f"| `{clip['stem']}` | {clip['camera']} | {clip['start_sec']:.1f}s | "
            f"{clip['stats']['n_raw_hits']}/{clip['stats']['n_frames']} | "
            f"{clip['stats']['emit_rate']:.2%} | {mean_txt} |"
        )
    return lines


def bestcam_table(payload: dict) -> list:
    lines = [
        "| Clip | Start | Selected raw hits | Selected emit rate | Mean selected conf | Top camera |",
        "|------|-------|-------------------|--------------------|--------------------|------------|",
    ]
    for clip in payload["clips"]:
        mean_txt = fmt_mean(clip["stats"].get("mean_selected_conf"))
        lines.append(
            f"| `{clip['stem']}` | {clip['start_sec']:.1f}s | "
            f"{clip['stats']['n_raw_hits']}/{clip['stats']['n_frames']} | "
            f"{clip['stats']['emit_rate']:.2%} | {mean_txt} | "
            f"{clip['stats']['top_camera']} |"
        )
    lines.append("")
    for clip in payload["clips"]:
        lines.append(f"**{clip['stem']}** {win_share_lines(clip)}")
        lines.append("")
    return lines


def win_share_lines(clip: dict) -> str:
    n = clip["stats"]["n_frames"] or 1
    parts = []
    for name, count in clip["stats"]["win_counts"].items():
        if count <= 0:
            continue
        parts.append(f"{name} {count}/{n} ({count / n:.1%})")
    return "Win share: " + ", ".join(parts)


def html_esc(text: str) -> str:
    return (
        str(text)
        .replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
    )


def ordered_clips(payload: dict) -> list:
    clips = list(payload.get("clips", []))
    if payload.get("layout") != "quad":
        return clips
    rank = {slot: i for i, slot in enumerate(QUAD_HTML_ORDER)}
    return sorted(clips, key=lambda c: rank.get(c.get("slot"), 99))


def write_html(out_dir: Path, payload: dict) -> Path:
    path = out_dir / "index.html"
    blocks = [html_header(payload)]
    if payload.get("layout") == "quad":
        blocks.append('<div class="quad">')
    for clip in ordered_clips(payload):
        blocks.append(html_clip(clip))
    if payload.get("layout") == "quad":
        blocks.append("</div>")
    blocks.append(html_filter_script())
    blocks.append("</main></body></html>")
    path.write_text("\n".join(blocks), encoding="utf-8")
    return path


def html_header(payload: dict) -> str:
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8"/>
<title>{html_esc(payload.get('title', '5x5 best-camera clips'))}</title>
<style>
body {{ font-family: ui-sans-serif, system-ui, sans-serif; background: #0b1220; color: #e8eefc; margin: 0; }}
main {{ max-width: 1280px; margin: 0 auto; padding: 28px 20px 64px; }}
h1 {{ margin: 0 0 8px; font-size: 28px; }}
.sub {{ color: #9db0d0; margin-bottom: 24px; }}
.clip {{ background: #151d2e; border-radius: 16px; padding: 18px; margin: 18px 0; }}
.meta {{ color: #9db0d0; font-size: 14px; margin-bottom: 12px; }}
.videos {{ display: grid; grid-template-columns: 1fr; gap: 12px; }}
video {{ width: 100%; border-radius: 10px; background: #000; }}
.bar {{ display: flex; height: 18px; border-radius: 9px; overflow: hidden; background: #0b1220; margin: 10px 0 6px; }}
.seg {{ height: 100%; }}
.legend {{ display: flex; flex-wrap: wrap; gap: 10px; font-size: 13px; color: #c5d3ea; }}
.filters {{ display: flex; flex-wrap: wrap; gap: 8px; margin: 12px 0; align-items: center; }}
.filters .label {{ color: #9db0d0; font-size: 13px; margin-right: 4px; }}
.filters button {{
  background: #0b1220; color: #e8eefc; border: 1px solid #334155;
  border-radius: 999px; padding: 6px 12px; cursor: pointer; font-size: 13px;
}}
.filters button.on {{ background: #2563eb; border-color: #2563eb; }}
.view-label {{ color: #9db0d0; font-size: 14px; margin: 0 0 8px; }}
.box-legend {{ color: #c5d3ea; font-size: 13px; margin: 0 0 10px; }}
.box-legend .emit {{ color: #4ade80; }}
.box-legend .raw {{ color: #fb923c; }}
.quad {{ display: block; }}
.quad .clip {{ margin: 18px 0; }}
table {{ width: 100%; border-collapse: collapse; font-size: 13px; margin-top: 10px; }}
th, td {{ text-align: left; padding: 6px 8px; border-bottom: 1px solid #243044; }}
tr[data-cam-filter] {{ cursor: pointer; }}
</style>
</head>
<body>
<main>
<h1>{html_esc(payload.get('title', '5x5 best-camera clips'))}</h1>
<p class="sub">Seed {html_esc(payload['seed'])} · {payload['n_clips']} clips ·
detect {payload['min_thr']} → emit {payload['emit_thresh']} ·
cameras {html_esc(payload.get('cameras'))} · select {html_esc(payload.get('select_camera'))}</p>
"""


def clip_cam_names(clip: dict) -> list:
    if clip.get("cameras"):
        return [row["camera"] for row in clip["cameras"]]
    return list(clip.get("stats", {}).get("per_camera", {}))


def filter_btn(key: str, label: str, src: str, on: bool = False) -> str:
    cls = "on" if on else ""
    return (
        f'<button type="button" class="{cls}" data-cam-filter="{html_esc(key)}" '
        f'data-src="{html_esc(src)}" data-label="{html_esc(label)}">{html_esc(label)}</button>'
    )


def html_cam_filter(clip: dict) -> str:
    stats = clip["stats"]
    stem = clip["stem"]
    overlay = f"overlay/{Path(stats['overlay']).name}"
    mosaic = f"overlay/{Path(stats['mosaic']).name}"
    buttons = [
        filter_btn("selected", "Selected", overlay, on=True),
        filter_btn("mosaic", "Mosaic", mosaic),
    ]
    for name in clip_cam_names(clip):
        buttons.append(filter_btn(name, name, f"overlay/{stem}_{name}_boxes.mp4"))
    return (
        '<div class="filters"><span class="label">Filter to 1 camera</span>'
        + "".join(buttons)
        + "</div>"
    )


def html_clip(clip: dict) -> str:
    stats = clip["stats"]
    overlay = Path(stats["overlay"]).name
    return f"""
<section class="clip">
<h2>{html_esc(clip.get('label') or clip['stem'])}</h2>
<p class="meta">{html_esc(fmt_clock(clip['start_sec']))}–{html_esc(fmt_clock(clip['start_sec'] + clip.get('duration_sec', 5)))} ·
selected raw {stats['n_raw_hits']}/{stats['n_frames']} ·
emit {stats['emit_rate']:.1%} · mean selected conf {html_esc(fmt_mean(stats.get('mean_selected_conf')))} ·
top camera {html_esc(stats['top_camera'])}</p>
{html_win_bar(stats)}
{html_cam_filter(clip)}
<div class="videos">
<p class="view-label">Watching: Selected</p>
<p class="box-legend"><span class="emit">green EMIT</span> = published ≥0.80 · <span class="raw">orange RAW</span> = detected, not emitted · marker is on the ball</p>
<video controls playsinline data-player type="video/mp4" src="overlay/{html_esc(overlay)}"></video>
</div>
{html_cam_table(clip)}
</section>
"""


def html_win_bar(stats: dict) -> str:
    n = stats["n_frames"] or 1
    colors = {
        "P1": "#60a5fa",
        "P6": "#34d399",
        "P7": "#fbbf24",
        "P8": "#f472b6",
        "P10": "#a78bfa",
        "P12": "#22d3ee",
        "Cam4plus": "#fb7185",
        "Cam5plus": "#4ade80",
        "none": "#475569",
    }
    segs = []
    legend = []
    for name, count in stats["win_counts"].items():
        if count <= 0:
            continue
        pct = 100.0 * count / n
        color = colors.get(name, "#64748b")
        segs.append(f'<div class="seg" style="width:{pct:.2f}%;background:{color}"></div>')
        legend.append(f"<span>{html_esc(name)} {count}/{n} ({pct:.1f}%)</span>")
    return f'<div class="bar">{"".join(segs)}</div><div class="legend">{"".join(legend)}</div>'


def html_cam_table(clip: dict) -> str:
    stats = clip["stats"]
    stem = clip["stem"]
    rows = [
        "<table><tr><th>Camera</th><th>Raw hits</th><th>Emit hold</th><th>Mean emit conf</th></tr>"
    ]
    for name, row in stats.get("per_camera", {}).items():
        src = f"overlay/{stem}_{name}_boxes.mp4"
        rows.append(
            f'<tr data-cam-filter="{html_esc(name)}" data-src="{html_esc(src)}" '
            f'data-label="{html_esc(name)}">'
            f"<td>{html_esc(name)}</td>"
            f"<td>{row['n_raw_hits']}/{row['n_frames']}</td>"
            f"<td>{row['emit_rate']:.1%}</td>"
            f"<td>{html_esc(fmt_mean(row.get('mean_emit_conf')))}</td>"
            "</tr>"
        )
    rows.append("</table>")
    return "".join(rows)


def html_filter_script() -> str:
    return """
<script>
function setClipView(clip, src, label, key) {
  const video = clip.querySelector("video[data-player]");
  const tag = clip.querySelector(".view-label");
  if (!video || !src) return;
  clip.querySelectorAll(".filters button").forEach((b) => {
    b.classList.toggle("on", b.getAttribute("data-cam-filter") === key);
  });
  if (tag) tag.textContent = "Watching: " + label;
  const t = video.currentTime || 0;
  const playing = !video.paused;
  video.src = src;
  video.addEventListener("loadedmetadata", function once() {
    video.removeEventListener("loadedmetadata", once);
    try { video.currentTime = Math.min(t, video.duration || t); } catch (e) {}
    if (playing) video.play();
  });
  video.load();
}
document.querySelectorAll(".clip").forEach((clip) => {
  clip.addEventListener("click", (ev) => {
    const btn = ev.target.closest("[data-cam-filter]");
    if (!btn || !clip.contains(btn)) return;
    setClipView(
      clip,
      btn.getAttribute("data-src"),
      btn.getAttribute("data-label") || btn.textContent,
      btn.getAttribute("data-cam-filter")
    );
  });
});
</script>
"""


def main() -> int:
    args = parse_args()
    if args.quad_test:
        args.select_camera = "max_conf"
        args.cameras = "all_match2"
        default_out = ROOT / "reports" / "eval_match2_v10" / PACK_NAME
        if args.out == default_out:
            args.out = ROOT / "reports" / "eval_match2_v10" / QUAD_PACK
    out = args.out
    out.mkdir(parents=True, exist_ok=True)
    if args.select_camera == "random":
        pack = run_random(args, out)
    else:
        pack = run_bestcam(args, out)
    payload = {
        "pack": out.name,
        "title": "4 quad test" if args.quad_test else "5x5 best-camera clips",
        "layout": "quad" if args.quad_test else "stack",
        "seed": args.seed,
        "n_clips": len(pack["clips"]),
        "clip_sec": args.clip_sec,
        "min_thr": args.min_thr,
        "emit_thresh": args.emit_thresh,
        "select_camera": args.select_camera,
        "cameras": args.cameras,
        "checkpoint": str(args.ball_checkpoint),
        "clips": pack["clips"],
    }
    summary = write_summary(out, payload)
    print(f"wrote {summary}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
