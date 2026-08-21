#!/usr/bin/env python3
"""Coach mosaic + Pitch 1 check video (default 60s match span).

Uses product ball map path: F0 fuse, MIN_SUPPORT from match3_xy, P10 hull, defish tiles.
"""
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[2]

from src.perception.rfdetr_local import LocalRFDETRDetector  # noqa: E402
from src.review.cam_mosaic import mosaic_quads_coach, match3_videos  # noqa: E402
from src.review.frame_sync import keep_top1_ball  # noqa: E402
from src.review.multicam_fuse import fuse_live_dets_for_pitch  # noqa: E402
from src.review.pitch1_panel import draw_pitch1_ball_panel  # noqa: E402
from src.review.team_live import TeamSession  # noqa: E402


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--start", type=int, default=2390)
    p.add_argument("--match-sec", type=float, default=60.0)
    p.add_argument("--src-fps", type=float, default=60.0)
    p.add_argument("--stride", type=int, default=15)
    p.add_argument("--out-fps", type=float, default=4.0)
    p.add_argument(
        "--out-dir",
        type=Path,
        default=ROOT / "reports/eval_match3/improve_eng_loop/phase1_check",
    )
    return p.parse_args()


def main() -> int:
    args = parse_args()
    out = args.out_dir
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

    sess = TeamSession()
    trail = []
    writer = None
    mp4 = out / "coach_mosaic_pitch_min.mp4"
    stats = []
    print(
        f"rendering n={len(frames)} out_fps={args.out_fps} "
        f"dur≈{len(frames) / args.out_fps:.1f}s match={args.match_sec}s",
        flush=True,
    )
    for i, fr in enumerate(frames):
        bag = {}
        mosaic = mosaic_quads_coach(
            vids,
            fr,
            tile_w=480,
            tile_h=270,
            dets_by_cam=bag,
            detect_fn=detect_fn,
            apply_defish=True,
        )
        live = fuse_live_dets_for_pitch(
            bag, apply_undistort=True, team_session=sess
        )
        players = live["players"]
        ball = live["ball_xy"]
        if ball is not None:
            trail.append(ball)
            trail = trail[-16:]
        teams = [int(p[2]) for p in players]
        n0, n1 = teams.count(0), teams.count(1)
        ng = sum(1 for t in teams if t < 0)
        pitch = draw_pitch1_ball_panel(
            280,
            mosaic.shape[0],
            ball_xy=ball,
            cam="live",
            mode=f"N blue={n0} red={n1} gray={ng}",
            trail=trail,
            players=players,
            tight=True,
        )
        if pitch.shape[0] != mosaic.shape[0]:
            pitch = cv2.resize(pitch, (pitch.shape[1], mosaic.shape[0]))
        combo = np.hstack([mosaic, pitch])
        t_s = (fr - args.start) / args.src_fps
        cv2.putText(
            combo,
            f"fr {fr}  t+{t_s:.1f}s  players={len(players)}  ball={'Y' if ball else 'N'}",
            (12, combo.shape[0] - 12),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 255, 255),
            2,
        )
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
            }
        )
        if (i + 1) % 10 == 0 or i == 0:
            print(f"{i + 1}/{len(frames)} fr={fr}", flush=True)
    writer.release()
    dur_s = len(frames) / args.out_fps
    ball_frac = sum(1 for s in stats if s["ball"]) / max(len(stats), 1)
    meta = {
        "ts": datetime.now(timezone.utc).isoformat(),
        "path": str(mp4.relative_to(ROOT)),
        "note": "Product ball map: undistort + P10 hull + MIN_SUPPORT 0.20 + F0 fuse",
        "frames_src": frames,
        "n_out_frames": len(frames),
        "out_fps": args.out_fps,
        "duration_s": round(dur_s, 2),
        "match_span_s": round(n_match / args.src_fps, 2),
        "stride": args.stride,
        "ball_frame_frac": round(ball_frac, 3),
        "stats": stats,
    }
    (out / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print("WROTE", mp4, f"dur={dur_s:.1f}s ball_frac={ball_frac:.2f}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
