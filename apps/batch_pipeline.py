# The Orchestrator
from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Optional

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import cv2
import numpy as np
import pandas as pd
import yaml
from dotenv import load_dotenv

from src.export.schema import frame_data_to_csv_row, get_csv_schema
from src.events.event_manager import EventManager
from src.events.events import EventDetector
from src.ingest.video import open_video_file
from src.mapping.mapping import PitchMapper
from src.mapping.match3_xy import (
    EMIT_CONF,
    apply_H,
    calib_undistort_params,
    fuse_balls_with_hold,
    load_calib_for_video,
    map_ball_box,
    scale_px,
    undistort_px,
)
from src.mapping.pitch_bounds import in_pitch_bounds
from src.perception.camera import detect_scene_cut, is_gameplay_view
from src.perception.rfdetr_local import build_detector
from src.perception.team_tracklet import TrackletAccumulator, TrackletTeamModel
from src.perception.track_ball import create_ball_tracker_wrapper
from src.perception.tracker import Tracker
from src.state.types import Ball, FrameData, Location, Player


def load_config(config_path: str = "configs/default.yaml") -> dict:
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def run_output_dir(base_dir: str, video_path: str) -> tuple[str, Path]:
    match_id = Path(video_path).stem
    run_dir = Path(base_dir) / match_id
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "checkpoints").mkdir(parents=True, exist_ok=True)
    return match_id, run_dir


def _release_accel_memory():
    """Drop cached MPS/CUDA tensors between ticks (Mac long-run stability)."""
    import gc

    gc.collect()
    try:
        import torch

        if hasattr(torch, "mps") and torch.backends.mps.is_available():
            torch.mps.empty_cache()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass


def map_player_box(pitch_mapper: PitchMapper, calib, bbox, frame_wh) -> Location:
    """Player center → pitch. Match 3: scale + optional Brown undistort + H (no hull gate)."""
    if calib is None:
        return pitch_mapper.bbox_center_to_pitch(bbox)
    x, y, w, h = bbox
    cx, cy = x + w * 0.5, y + h * 0.5
    wh = frame_wh or calib.get("image_wh") or [1920, 1080]
    px, py = scale_px(cx, cy, wh, calib.get("image_wh") or wh)
    params = calib_undistort_params(calib)
    if params:
        cw, ch = calib.get("image_wh") or wh
        px, py = undistort_px(px, py, cw, ch, params)
    xy = apply_H(calib["H"], px, py)
    if xy is None:
        return pitch_mapper.bbox_center_to_pitch(bbox)
    return Location(x=xy[0], y=xy[1])


def map_ball_detection(pitch_mapper: PitchMapper, calib, bbox, conf, frame_wh) -> Optional[Location]:
    if calib is not None:
        mapped = map_ball_box(calib, bbox, conf=conf, frame_wh=frame_wh)
        if mapped is None:
            return None
        return Location(x=mapped["xy"][0], y=mapped["xy"][1])
    location = pitch_mapper.bbox_foot_to_pitch(bbox)
    if pitch_mapper.homography is not None:
        if not in_pitch_bounds(location.x, location.y, margin_m=1.0):
            return None
    return location


def collect_ball_maps(calib, detections, ball_id: int, frame_wh) -> list:
    """Product Match 3 path: map all ball dets (fuse/hold applies EMIT_CONF)."""
    rows = []
    for det in detections:
        if int(det.class_id) != int(ball_id):
            continue
        hit = map_ball_box(calib, det.bbox, float(det.confidence), frame_wh=frame_wh)
        if hit is not None:
            rows.append(hit)
    return rows


def apply_match3_ball_hold(
    frame_data: FrameData,
    calib,
    ball_id: int,
    frame_wh,
    prev_emit: Optional[dict],
    frames_since_emit: int,
) -> tuple[FrameData, Optional[dict], int]:
    """Override tracker ball with map_ball_box + F0 hold (same as M1 product)."""
    rows = collect_ball_maps(calib, frame_data.detections or [], ball_id, frame_wh)
    emit = fuse_balls_with_hold(prev_emit, rows, frames_since_emit)
    if emit is None:
        frame_data.ball = None
        return frame_data, prev_emit, frames_since_emit + 1
    frame_data.ball = Ball(
        x_pitch=float(emit["xy"][0]),
        y_pitch=float(emit["xy"][1]),
        bbox=(0.0, 0.0, 1.0, 1.0),
        frame_id=frame_data.frame_id,
        timestamp=frame_data.timestamp,
        object_id=-1,
    )
    return frame_data, emit, 0


def _cam_id_from_video(video_path: str) -> str | None:
    try:
        _gs = _ROOT / "scripts" / "gold_set"
        if str(_gs) not in sys.path:
            sys.path.insert(0, str(_gs))
        from raw_cam_id import cam_id_from_raw_name  # noqa: WPS433

        return cam_id_from_raw_name(Path(video_path).name)
    except Exception:
        return None


def golden_batch_pass(
    cap,
    detector,
    tracker: Tracker,
    pitch_mapper: PitchMapper,
    config: dict,
    calib,
    cam_id: str | None,
    golden_frames: int,
    start_frame: int,
    fps: float,
) -> TrackletTeamModel:
    """Pass 1: accumulate tracklet jersey features, fit match centroids."""
    acc = TrackletAccumulator()
    model = TrackletTeamModel()
    player_id = config["detection"]["player_class_id"]
    prev_frame = None
    frame_id = start_frame
    end_frame = start_frame + golden_frames
    collected = 0

    while frame_id < end_frame:
        ret, frame = cap.read()
        if not ret:
            break
        camera_cfg = config.get("camera") or {}
        if not is_gameplay_view(frame, green_threshold=float(camera_cfg.get("green_threshold", 0.5))):
            prev_frame = frame.copy()
            frame_id += 1
            continue
        if detect_scene_cut(frame, prev_frame, threshold=float(camera_cfg.get("scene_cut_threshold", 0.7))):
            tracker.reset()
        detections = detector.detect(frame)
        if detections:
            tracked = tracker.update(detections, frame)
            frame_wh = (frame.shape[1], frame.shape[0])
            for obj in tracked:
                det = obj.detection
                if int(det.class_id) != int(player_id):
                    continue
                loc = map_player_box(pitch_mapper, calib, det.bbox, frame_wh)
                acc.add(
                    obj.object_id,
                    frame,
                    det.bbox,
                    pitch_xy=(loc.x, loc.y),
                    cam=cam_id,
                    frame_wh=frame_wh,
                )
                collected += 1
        prev_frame = frame.copy()
        frame_id += 1

    min_tr = int(config.get("team_assignment", {}).get("min_tracklets", 15))
    if model.fit_from_accumulator(acc, min_tracklets=min_tr):
        print(f"Golden Batch: {len(acc.tracklet_medians())} tracklets, {collected} crops")
    else:
        print(f"Golden Batch fit failed ({collected} crops) — teams will be gray")
    return model


def process_frame(
    frame: np.ndarray,
    frame_id: int,
    timestamp: float,
    detector,
    tracker: Tracker,
    pitch_mapper: PitchMapper,
    prev_frame: Optional[np.ndarray],
    config: dict,
    calib=None,
    team_model: TrackletTeamModel | None = None,
    cam_id: str | None = None,
) -> Optional[FrameData]:
    camera_cfg = config.get("camera") or {}
    if not is_gameplay_view(frame, green_threshold=float(camera_cfg.get("green_threshold", 0.5))):
        return None

    if detect_scene_cut(frame, prev_frame, threshold=float(camera_cfg.get("scene_cut_threshold", 0.7))):
        tracker.reset()

    detections = detector.detect(frame)
    if not detections:
        return None

    tracked_objects = tracker.update(detections, frame)
    if not tracked_objects:
        return None

    frame_wh = (frame.shape[1], frame.shape[0])
    player_id = config["detection"]["player_class_id"]
    pitch_positions: dict[int, tuple[float, float]] = {}
    for obj in tracked_objects:
        det = obj.detection
        if int(det.class_id) == int(player_id):
            loc = map_player_box(pitch_mapper, calib, det.bbox, frame_wh)
            pitch_positions[int(obj.object_id)] = (loc.x, loc.y)

    if team_model is not None and team_model.centroids is not None:
        tracked_objects = team_model.apply_to_tracked(
            tracked_objects,
            frame,
            player_class_id=player_id,
            pitch_positions=pitch_positions,
            cam=cam_id,
            frame_wh=frame_wh,
        )

    ball_id = config["detection"]["ball_class_id"]
    players = []
    ball = None

    for obj in tracked_objects:
        det = obj.detection
        if det.class_id == player_id:
            location = map_player_box(pitch_mapper, calib, det.bbox, frame_wh)
            players.append(
                Player(
                    object_id=obj.object_id,
                    team_id=obj.team_id if obj.team_id is not None else -1,
                    x_pitch=location.x,
                    y_pitch=location.y,
                    bbox=det.bbox,
                    frame_id=frame_id,
                    timestamp=timestamp,
                )
            )
        elif det.class_id == ball_id and calib is None:
            # Match 3 calib → ball filled later via map + hold (product path).
            location = map_ball_detection(
                pitch_mapper, calib, det.bbox, float(det.confidence), frame_wh
            )
            if location is None:
                continue
            ball = Ball(
                x_pitch=location.x,
                y_pitch=location.y,
                bbox=det.bbox,
                frame_id=frame_id,
                timestamp=timestamp,
                object_id=obj.object_id,
            )

    if not players:
        return None

    return FrameData(
        frame_id=frame_id,
        timestamp=timestamp,
        players=players,
        ball=ball,
        detections=detections,
    )


def process_video(
    video_path: str,
    config: dict,
    output_dir: str = "data/output",
    max_frames: Optional[int] = None,
    start_frame: int = 0,
):
    load_dotenv()
    match_id, run_dir = run_output_dir(output_dir, video_path)
    detector = build_detector(config)

    pitch_mapper = PitchMapper(
        pitch_length=config["mapping"]["pitch_length"],
        pitch_width=config["mapping"]["pitch_width"],
    )
    calib = load_calib_for_video(video_path)
    if calib is not None:
        pitch_mapper.set_homography(calib["H"])
        print(f"Match 3 calib loaded for {calib.get('camera')} (ball = map + hold)")

    ev_cfg = config["events"]
    event_detector = EventDetector(
        pitch_mapper=pitch_mapper,
        pass_velocity_threshold=ev_cfg["pass_velocity_threshold"],
        dribble_distance_threshold=ev_cfg["dribble_distance_threshold"],
        shot_velocity_threshold=ev_cfg["shot_velocity_threshold"],
        recovery_proximity=ev_cfg["recovery_proximity"],
        emit_conf=float(ev_cfg.get("emit_conf", 0.80)),
        half_length_m=float(ev_cfg.get("half_length_m", 26.95)),
        shot_goal_band_m=float(ev_cfg.get("shot_goal_band_m", 5.0)),
        enable_dribble=bool(ev_cfg.get("enable_dribble", True)),
        enable_movement=bool(ev_cfg.get("enable_movement", True)),
        movement_velocity_min=float(ev_cfg.get("movement_velocity_min", 1.0)),
        movement_proximity=float(ev_cfg.get("movement_proximity", 4.0)),
        co_move_min_player_m=float(ev_cfg.get("co_move_min_player_m", 0.15)),
        co_move_min_cos=float(ev_cfg.get("co_move_min_cos", 0.55)),
        dribble_window_frames=int(ev_cfg.get("dribble_window_frames", 3)),
        dribble_min_carry_m=float(ev_cfg.get("dribble_min_carry_m", 0.6)),
        dribble_co_move_streak=int(ev_cfg.get("dribble_co_move_streak", 2)),
    )
    event_manager = EventManager(
        checkpoint_interval=config["checkpoint"]["interval_frames"],
        output_dir=str(run_dir),
    )

    cap = open_video_file(video_path)
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 30.0)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    start_frame = max(0, int(start_frame))
    if start_frame > 0:
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    remaining = max(0, frame_count - start_frame)
    frames_to_run = remaining if max_frames is None else min(remaining, max_frames)

    tracker_cfg = config.get("tracker", {})
    team_cfg = config.get("team_assignment") or {}
    golden_frames = int(team_cfg.get("golden_batch_frames", 600))
    cam_id = _cam_id_from_video(video_path)

    def _make_tracker():
        base = Tracker(
            track_thresh=tracker_cfg.get("track_thresh", 0.10),
            high_thresh=tracker_cfg.get("high_thresh", 0.35),
            track_buffer=tracker_cfg.get("track_buffer", 30),
            match_thresh=tracker_cfg.get("match_thresh", 0.8),
            frame_rate=int(round(fps)) if fps > 1 else tracker_cfg.get("frame_rate", 30),
            emit_thresh=tracker_cfg.get("emit_thresh", EMIT_CONF),
            ema_alpha=tracker_cfg.get("ema_alpha", 0.3),
            apply_emit_gate=tracker_cfg.get("apply_emit_gate", True),
        )
        return create_ball_tracker_wrapper(base, min_track_length=5, fit_threshold=0.15)

    gb_tracker = _make_tracker()
    cap_gb = open_video_file(video_path)
    if start_frame > 0:
        cap_gb.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    gb_len = min(golden_frames, frames_to_run)
    team_model = golden_batch_pass(
        cap_gb,
        detector,
        gb_tracker,
        pitch_mapper,
        config,
        calib,
        cam_id,
        gb_len,
        start_frame,
        fps,
    )
    cap_gb.release()
    centroids_path = run_dir / "team_centroids.json"
    team_model.save(centroids_path)
    if team_model.centroids is not None:
        print(f"Team centroids → {centroids_path}")

    tracker = _make_tracker()
    print(
        f"Processing {match_id}: {frames_to_run} frames "
        f"(start={start_frame}, total={frame_count}) @ {fps} fps → {run_dir}"
    )

    prev_frame = None
    prev_frame_data = None
    frame_id = start_frame
    end_frame = start_frame + frames_to_run
    csv_rows = []
    prev_ball_emit = None
    frames_since_ball = 10**9
    ball_emit_frames = 0
    ball_id = config["detection"]["ball_class_id"]

    while frame_id < end_frame:
        ret, frame = cap.read()
        if not ret:
            break

        timestamp = frame_id / fps if fps > 0 else frame_id * 0.033
        frame_data = process_frame(
            frame,
            frame_id,
            timestamp,
            detector,
            tracker,
            pitch_mapper,
            prev_frame,
            config,
            calib=calib,
            team_model=team_model,
            cam_id=cam_id,
        )
        if frame_data:
            if calib is not None:
                frame_wh = (frame.shape[1], frame.shape[0])
                frame_data, prev_ball_emit, frames_since_ball = apply_match3_ball_hold(
                    frame_data,
                    calib,
                    ball_id,
                    frame_wh,
                    prev_ball_emit,
                    frames_since_ball,
                )
            if frame_data.ball is not None:
                ball_emit_frames += 1
            events = event_detector.detect_events(frame_data, prev_frame_data)
            if events:
                event_manager.add_events(events)
            csv_rows.extend(frame_data_to_csv_row(frame_data))
            prev_frame_data = frame_data
        elif calib is not None:
            frames_since_ball += 1

        event_manager.tick_frame(frame_id)
        prev_frame = frame.copy()
        frame_id += 1
        done = frame_id - start_frame
        if done % 50 == 0:
            print(
                f"Processed {done}/{frames_to_run} frames "
                f"(ball_emit_frames={ball_emit_frames})"
            )
            _release_accel_memory()

    cap.release()
    _release_accel_memory()

    events_csv = run_dir / "events.csv"
    events_json = run_dir / "events.json"
    frame_csv = run_dir / "frame_data.csv"
    event_manager.save_final_output(
        match_id=match_id,
        csv_path=str(events_csv),
        json_path=str(events_json),
    )
    if csv_rows:
        pd.DataFrame(csv_rows, columns=get_csv_schema()).to_csv(frame_csv, index=False)
        print(f"Saved frame data → {frame_csv}")
    print(f"Saved events → {events_csv} / {events_json}")
    print(f"Ball emit frames: {ball_emit_frames}/{frame_id - start_frame}")
    print("Processing complete!")
    return str(run_dir)


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Soccer Analysis Pipeline")
    parser.add_argument("--video", type=str, required=True, help="Path to input video")
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    parser.add_argument("--output", type=str, default="data/output")
    parser.add_argument("--max-frames", type=int, default=None)
    parser.add_argument("--start-frame", type=int, default=0, help="Seek before processing")
    args = parser.parse_args()
    config = load_config(args.config)
    process_video(
        args.video,
        config,
        args.output,
        max_frames=args.max_frames,
        start_frame=args.start_frame,
    )


if __name__ == "__main__":
    main()
