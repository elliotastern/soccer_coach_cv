#!/usr/bin/env python3
"""Coach mosaic + Pitch 1 check video (default 60s match span).

Uses product ball map path: F0 fuse, MIN_SUPPORT from match3_xy, P10 hull, defish tiles.
"""
from __future__ import annotations

import argparse
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[2]
import sys

sys.path.insert(0, str(ROOT))

from src.events.events import EventDetector  # noqa: E402
from src.perception.rfdetr_local import LocalRFDETRDetector  # noqa: E402
from src.review.cam_mosaic import (
    QUAD_ROTATE_180,
    compose_coach_stack,
    mosaic_grid_size,
    mosaic_quads_coach,
    pitch_stack_metrics,
    match3_videos,
    _tile,
    _is_ball_det,
)  # noqa: E402
from src.review.frame_sync import keep_top1_ball  # noqa: E402
from src.mapping.fuse_config import load_fuse_config  # noqa: E402
from src.review.cam_mosaic import fill_fuse_cams_for_pitch  # noqa: E402
from src.review.multicam_fuse import fuse_live_dets_for_pitch  # noqa: E402
from src.review.pitch1_panel import draw_pitch1_ball_panel  # noqa: E402
from src.perception.team_strategy import session_from_config  # noqa: E402
from src.state.types import Ball, FrameData, Player  # noqa: E402

EVENT_BAR_H = 56
EVENT_COLORS = {
    "pass": (80, 180, 255),
    "shot": (60, 60, 255),
    "recovery": (80, 220, 120),
    "dribble": (200, 160, 80),
    "movement": (180, 180, 180),
}


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--start", type=int, default=2390)
    p.add_argument("--match-sec", type=float, default=60.0)
    p.add_argument("--src-fps", type=float, default=60.0)
    p.add_argument("--stride", type=int, default=15)
    p.add_argument("--out-fps", type=float, default=4.0)
    p.add_argument("--events-bar", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument(
        "--debug-cam",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Label pitch dots with source cam (P7–P10); faint dots = pre-fuse maps",
    )
    p.add_argument(
        "--out-dir",
        type=Path,
        default=ROOT / "reports/eval_match3/improve_eng_loop/phase1_check",
    )
    p.add_argument(
        "--out-file",
        type=str,
        default="coach_mosaic_pitch_min.mp4",
        help="Output mp4 name inside --out-dir",
    )
    p.add_argument(
        "--events-only",
        action="store_true",
        help="Re-run fuse+heuristic emits only (no mosaic MP4); updates emits_render.json",
    )
    p.add_argument(
        "--layout",
        choices=("mosaic", "best_ball"),
        default="mosaic",
        help="mosaic = quad + pitch; best_ball = single best-ball cam + pitch",
    )
    default_kit = ROOT / "data/output/match_4_5min/P10-match4/team_centroids.json"
    p.add_argument(
        "--kit-centroids",
        type=Path,
        default=None,
        help=f"Pre-labeled team_centroids.json (auto: {default_kit.name} if present)",
    )
    p.add_argument(
        "--fuse-mode",
        choices=("pitch_merge", "triangulate_3d"),
        default=None,
        help="Override configs/default.yaml fuse.mode",
    )
    p.add_argument(
        "--fuse-ukf",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Override configs/default.yaml fuse.ukf_enabled",
    )
    p.add_argument(
        "--fuse-cams",
        choices=("quad", "all"),
        default=None,
        help="Override configs/default.yaml fuse.cams (all = eight cams in pitch fuse bag)",
    )
    return p.parse_args()


def _live_to_frame(fr: int, t_s: float, live: dict) -> FrameData:
    players = []
    for p in live["players"]:
        x, y = float(p[0]), float(p[1])
        team = int(p[2]) if len(p) > 2 else -1
        pid = int(p[3]) if len(p) > 3 else 0
        players.append(Player(pid, team, x, y, (0, 0, 10, 10), fr, t_s))
    ball = None
    if live.get("ball_xy") is not None:
        bx, by = live["ball_xy"]
        ball = Ball(float(bx), float(by), (0, 0, 4, 4), fr, t_s)
    return FrameData(fr, t_s, players, ball)


def draw_events_bar(width: int, t_s: float, recent: list[dict], flash: str | None) -> np.ndarray:
    bar = np.zeros((EVENT_BAR_H, width, 3), dtype=np.uint8)
    bar[:] = (28, 28, 32)
    cv2.putText(
        bar,
        f"events  t+{t_s:.1f}s",
        (10, 22),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (220, 220, 220),
        1,
    )
    x = 160
    for em in recent[-6:]:
        label = f"{em['type']} @{em['t_end']:.1f}s"
        col = EVENT_COLORS.get(em["type"], (200, 200, 200))
        if flash and em["type"] == flash and abs(t_s - em["t_end"]) < 0.35:
            col = tuple(min(255, c + 60) for c in col)
        cv2.rectangle(bar, (x, 28), (x + 118, 50), col, -1)
        cv2.putText(
            bar, label[:14], (x + 4, 45), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (20, 20, 20), 1
        )
        x += 124
    if flash and not any(abs(t_s - em["t_end"]) < 0.35 for em in recent[-3:]):
        cv2.putText(
            bar,
            f"NEW {flash.upper()}",
            (width - 180, 45),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            EVENT_COLORS.get(flash, (255, 255, 255)),
            2,
        )
    return bar


def pick_best_ball_cam(bag: dict) -> str:
    for cam in ("P10", "P9", "P7", "P8"):
        if any(_is_ball_det(d) for d in (bag.get(cam) or [])):
            return cam
    return "P10"


def render_cam_panel(
    layout: str,
    vids,
    fr: int,
    bag: dict,
    detect_fn,
    apply_defish: bool,
    tile_w: int,
    tile_h: int,
    grid_w: int,
    grid_h: int,
) -> np.ndarray:
    if layout == "mosaic":
        return mosaic_quads_coach(
            vids,
            fr,
            tile_w=tile_w,
            tile_h=tile_h,
            dets_by_cam=bag,
            detect_fn=detect_fn,
            apply_defish=apply_defish,
        )
    mosaic_quads_coach(
        vids,
        fr,
        tile_w=tile_w,
        tile_h=tile_h,
        dets_by_cam=bag,
        detect_fn=detect_fn,
        apply_defish=apply_defish,
    )
    best = pick_best_ball_cam(bag)
    rotate = best in QUAD_ROTATE_180
    tile = _tile(
        vids,
        best,
        fr,
        tile_w * 2,
        tile_h * 2,
        dets_by_cam=bag,
        detect_fn=detect_fn,
        rotate_180=rotate,
        apply_defish=apply_defish,
    )
    panel = cv2.resize(tile, (grid_w, grid_h), interpolation=cv2.INTER_AREA)
    cv2.rectangle(panel, (0, 0), (panel.shape[1] - 1, panel.shape[0] - 1), (0, 200, 255), 2)
    cv2.putText(
        panel,
        f"BEST BALL · {best}",
        (12, 28),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.75,
        (0, 220, 255),
        2,
    )
    return panel


def h264_encode(src: Path, fps: float) -> None:
    """QuickTime/Cursor preview need yuv420p H.264; preserve fps for smooth scrub."""
    tmp = src.with_suffix(".h264.mp4")
    cmd = [
        "ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
        "-i", str(src), "-r", f"{fps:.3f}", "-c:v", "libx264", "-pix_fmt", "yuv420p",
        "-movflags", "+faststart", str(tmp),
    ]
    subprocess.run(cmd, check=True)
    tmp.replace(src)


def main() -> int:
    args = parse_args()
    out = args.out_dir if args.out_dir.is_absolute() else (ROOT / args.out_dir)
    out = out.resolve()
    out.mkdir(parents=True, exist_ok=True)
    n_match = int(round(args.match_sec * args.src_fps))
    end = args.start + n_match - 1
    frames = list(range(args.start, end + 1, args.stride))
    vids = match3_videos(ROOT)
    det = LocalRFDETRDetector(
        player_checkpoint=str(ROOT / "models/people_after_100_epochs.pth"),
        ball_checkpoint=str(ROOT / "models/v12_hard_snaps/post_train/checkpoint.pth"),
        confidence_threshold=0.15,
        enhance_ball=False,
        use_sahi=False,
        use_kalman=False,
        player_nms_iou=0.30,
        ball_nms_iou=0.4,
    )

    def detect_fn(cam, frame_bgr):
        return keep_top1_ball(det.detect(frame_bgr))

    cfg_path = ROOT / "configs/default.yaml"
    cfg = {}
    if cfg_path.is_file():
        cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8")) or {}
    sess = session_from_config(cfg)
    kit_path = args.kit_centroids
    if kit_path is None:
        auto_kit = ROOT / "data/output/match_4_5min/P10-match4/team_centroids.json"
        if auto_kit.is_file():
            kit_path = auto_kit
    if kit_path is not None and Path(kit_path).is_file():
        if sess.load_centroids_file(Path(kit_path)):
            print(f"kit centroids loaded from {kit_path}", flush=True)
    event_det = EventDetector() if args.events_bar else None
    prev_fd = None
    recent_emits: list[dict] = []
    all_emits: list[dict] = []
    trail = []
    writer = None
    mp4 = out / args.out_file
    stats = []
    events_only = bool(args.events_only)
    fuse_cfg = load_fuse_config(cfg_path)
    if args.fuse_mode is not None:
        fuse_cfg = {**fuse_cfg, "mode": args.fuse_mode}
    if args.fuse_ukf is not None:
        fuse_cfg = {**fuse_cfg, "ukf_enabled": bool(args.fuse_ukf)}
    if args.fuse_cams is not None:
        fuse_cfg = {**fuse_cfg, "cams": args.fuse_cams}
    ball_ukf = None
    use_full_fuse_bag = (
        fuse_cfg.get("cams") == "all" or fuse_cfg.get("mode") == "triangulate_3d"
    )
    print(
        f"{'events-only' if events_only else 'rendering'} n={len(frames)} "
        f"out_fps={args.out_fps} dur≈{len(frames) / args.out_fps:.1f}s match={args.match_sec}s",
        flush=True,
    )
    apply_defish = True
    # Product pairing: defish tiles → detect on shown pixels → map WITHOUT second undistort.
    # Review app: apply_undistort=not apply_defish. Never set both True for P7–P10.
    apply_undistort = not apply_defish
    assert not (apply_defish and apply_undistort), (
        "defish+undistort double-warps P7–P10 feet; use apply_undistort=not apply_defish"
    )
    tile_w, tile_h = 480, 270
    grid_w, grid_h = mosaic_grid_size(tile_w, tile_h)
    stack = pitch_stack_metrics(grid_w, grid_h, drop_top=True, scale=0.46)
    for i, fr in enumerate(frames):
        bag = {}
        if events_only:
            mosaic_quads_coach(
                vids,
                fr,
                tile_w=tile_w,
                tile_h=tile_h,
                dets_by_cam=bag,
                detect_fn=detect_fn,
                apply_defish=apply_defish,
            )
        else:
            mosaic = render_cam_panel(
                args.layout,
                vids,
                fr,
                bag,
                detect_fn,
                apply_defish,
                tile_w,
                tile_h,
                grid_w,
                grid_h,
            )
        fuse_bag: dict = {}
        if use_full_fuse_bag:
            fill_fuse_cams_for_pitch(
                vids,
                fr,
                fuse_bag,
                detect_fn,
                apply_defish,
                fuse_cfg=fuse_cfg,
                single_ball=False,
            )
        else:
            fuse_bag = bag
        live = fuse_live_dets_for_pitch(
            fuse_bag,
            apply_undistort=apply_undistort,
            team_session=sess,
            debug_cam=args.debug_cam,
            fuse_cfg=fuse_cfg,
            ukf_state=ball_ukf,
        )
        ball_ukf = live.get("ukf_state")
        players = live["players"]
        ball = live["ball_xy"]
        if ball is not None:
            trail.append(ball)
            trail = trail[-16:]
        teams = [int(p[2]) for p in players]
        n0, n1 = teams.count(0), teams.count(1)
        ng = sum(1 for t in teams if t < 0)
        t_s = (fr - args.start) / args.src_fps
        flash = None
        if event_det is not None:
            fd = _live_to_frame(fr, t_s, live)
            if prev_fd is not None:
                for ev in event_det.detect_events(fd, prev_fd):
                    row = {
                        "type": ev.type.value,
                        "t_end": round(ev.timestamp_end, 3),
                        "t_start": round(ev.timestamp_start, 3),
                        "confidence": round(ev.confidence, 3),
                        "frame_id": fr,
                    }
                    recent_emits.append(row)
                    all_emits.append(row)
                    flash = ev.type.value
            prev_fd = fd
        if not events_only:
            pitch = draw_pitch1_ball_panel(
                stack["panel_w"],
                stack["panel_h"],
                ball_xy=ball,
                cam="live",
                mode=f"N blue={n0} red={n1} gray={ng}",
                trail=trail,
                players=players,
                player_cams=live.get("player_cams", ()),
                raw_player_maps=live.get("player_maps_all", ()),
                tight=True,
                orient_hints=True,
                field_w=stack["field_w"],
                field_h=stack["field_h"],
                band_w=stack["band_w"],
                drop_top=True,
                map_orient=stack["map_orient"],
            )
            combo = compose_coach_stack(mosaic, pitch, connect=True)
            cv2.putText(
                combo,
                f"fr {fr}  t+{t_s:.1f}s  players={len(players)}  ball={'Y' if ball else 'N'}",
                (12, combo.shape[0] - 12),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (255, 255, 255),
                2,
            )
            if args.events_bar:
                bar = draw_events_bar(combo.shape[1], t_s, recent_emits, flash)
                combo = np.vstack([combo, bar])
            if writer is None:
                h, w = combo.shape[:2]
                writer = cv2.VideoWriter(
                    str(mp4), cv2.VideoWriter_fourcc(*"mp4v"), args.out_fps, (w, h)
                )
            writer.write(combo)
            if i == 0:
                cv2.imwrite(str(out / "still_first.jpg"), combo)
            if i == len(frames) // 2:
                cv2.imwrite(str(out / "still_mid.jpg"), combo)
            if i == len(frames) - 1:
                cv2.imwrite(str(out / "still_last.jpg"), combo)
        stats.append(
            {
                "fr": fr,
                "n": len(players),
                "n0": n0,
                "n1": n1,
                "gray": ng,
                "ball": ball is not None,
                "event_flash": flash,
            }
        )
        if (i + 1) % 10 == 0 or i == 0:
            print(f"{i + 1}/{len(frames)} fr={fr} emits={len(all_emits)}", flush=True)
    if writer is not None:
        writer.release()
        h264_encode(mp4, args.out_fps)
    dur_s = len(frames) / args.out_fps
    ball_frac = sum(1 for s in stats if s["ball"]) / max(len(stats), 1)
    meta_path = out / "meta.json"
    if events_only and meta_path.is_file():
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        meta["emits"] = all_emits
        meta["n_emits"] = len(all_emits)
        meta["ts"] = datetime.now(timezone.utc).isoformat()
        note = str(meta.get("note", ""))
        meta["note"] = note + "; emits rebuilt (shot gates v2)"
    else:
        meta = {
            "ts": datetime.now(timezone.utc).isoformat(),
            "path": str(mp4.relative_to(ROOT)),
            "note": (
                "Product map: defish tiles, apply_undistort=not apply_defish, "
                "map_player_box (P8 lower-zone H fallback), P_Goal2 hull, F0 fuse; "
                "events bar = heuristic pass/shot/recovery on fused xy"
            ),
            "layout": args.layout,
            "debug_cam": bool(args.debug_cam),
            "events_bar": bool(args.events_bar),
            "n_emits": len(all_emits),
            "emits": all_emits,
            "apply_defish": True,
            "apply_undistort": False,
            "frames_src": frames,
            "n_out_frames": len(frames),
            "out_fps": args.out_fps,
            "duration_s": round(dur_s, 2),
            "match_span_s": round(n_match / args.src_fps, 2),
            "stride": args.stride,
            "ball_frame_frac": round(ball_frac, 3),
            "stats": stats,
        }
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    if all_emits:
        (out / "emits_render.json").write_text(
            json.dumps({"emits": all_emits}, indent=2), encoding="utf-8"
        )
    if events_only:
        print(
            "WROTE emits_render.json",
            f"n={len(all_emits)} (events-only; MP4 unchanged)",
            flush=True,
        )
    else:
        print(
            "WROTE", mp4, f"dur={dur_s:.1f}s ball_frac={ball_frac:.2f} emits={len(all_emits)}",
            flush=True,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
