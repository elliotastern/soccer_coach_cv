#!/usr/bin/env python3
"""Unit checks for 5x5 clip naming and window picking (no model)."""
from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts" / "gold_set"))

from eval_match2_v10_video_system import pick_selected
from run_5x5_ball_clips import (
    CLIP_SEC,
    MATCH2_CAMS,
    N_CLIPS,
    clip_stem,
    html_clip,
    named_quad_windows,
    paint_view,
    ordered_clips,
    parse_clock,
    pick_random_windows,
    pick_synced_windows,
    windows_overlap,
    write_html,
)


def test_clip_stem_uses_5x5_name():
    stem = clip_stem(1, {"camera": "Cam5plus", "start_sec": 33.4})
    assert stem.startswith("5x5_clip_01_")
    assert "Cam5plus" in stem
    assert "t00033.4s" in stem


def test_clip_stem_bestcam():
    stem = clip_stem(2, {"camera": "bestcam", "start_sec": 70.4})
    assert stem == "5x5_clip_02_bestcam_t00070.4s"


def test_overlap_same_camera():
    taken = [{"camera": "Cam5plus", "start_sec": 10.0}]
    assert windows_overlap("Cam5plus", 12.0, 5.0, taken) is True
    assert windows_overlap("Cam5plus", 20.0, 5.0, taken) is False
    assert windows_overlap("Cam4plus", 12.0, 5.0, taken) is False


def test_pick_five_five_second_windows():
    fake = {"Cam5plus": 1200.0, "Cam4plus": 1200.0}
    with patch("run_5x5_ball_clips.video_duration_sec", side_effect=lambda path: fake["Cam5plus"]):
        windows = pick_random_windows(N_CLIPS, CLIP_SEC, seed=20260813)
    assert len(windows) == 5
    starts_by_cam = {}
    for row in windows:
        assert row["duration_sec"] == 5.0
        assert row["camera"] in ("Cam5plus", "Cam4plus")
        assert 0.0 <= row["start_sec"] <= 1200.0 - 5.0
        starts_by_cam.setdefault(row["camera"], []).append(row["start_sec"])
    for starts in starts_by_cam.values():
        ordered = sorted(starts)
        for a, b in zip(ordered, ordered[1:]):
            assert abs(b - a) >= 5.0


def test_pick_synced_windows_same_start():
    with patch("run_5x5_ball_clips.video_duration_sec", return_value=1200.0):
        windows = pick_synced_windows(
            {"n": N_CLIPS, "clip_sec": CLIP_SEC, "seed": 20260813},
            MATCH2_CAMS,
        )
    assert len(windows) == 5
    names = [name for name, _path in MATCH2_CAMS]
    assert len(names) == 8
    starts = []
    for row in windows:
        assert row["camera"] == "bestcam"
        assert row["duration_sec"] == 5.0
        assert len(row["cameras"]) == 8
        assert [cam["camera"] for cam in row["cameras"]] == names
        starts.append(row["start_sec"])
    ordered = sorted(starts)
    for a, b in zip(ordered, ordered[1:]):
        assert abs(b - a) >= 5.0


def test_pick_selected_eight_cameras():
    pred_map = {name: [] for name, _path in MATCH2_CAMS}
    pred_map["P10"] = [([0, 0, 30, 30], 0.55, 30.0)]
    pred_map["P1"] = [([0, 0, 20, 20], 0.81, 20.0)]
    cam, pred = pick_selected(pred_map, "max_conf")
    assert cam == "P1"
    assert pred[1] == 0.81


def test_pick_selected_all_empty():
    pred_map = {name: [] for name, _path in MATCH2_CAMS}
    cam, pred = pick_selected(pred_map, "max_conf")
    assert cam is None
    assert pred is None


def test_html_clip_has_per_camera_filter():
    clip = {
        "stem": "5x5_clip_01_bestcam_t01285.1s",
        "start_sec": 1285.1,
        "duration_sec": 5.0,
        "cameras": [{"camera": "Cam4plus"}, {"camera": "Cam5plus"}],
        "stats": {
            "n_frames": 10,
            "n_raw_hits": 10,
            "emit_rate": 0.0,
            "mean_selected_conf": 0.6,
            "top_camera": "Cam4plus",
            "overlay": "overlay/sel.mp4",
            "mosaic": "overlay/mos.mp4",
            "win_counts": {"Cam4plus": 8, "Cam5plus": 2},
            "per_camera": {
                "Cam4plus": {
                    "n_raw_hits": 8, "n_frames": 10,
                    "emit_rate": 0.0, "mean_emit_conf": None,
                },
                "Cam5plus": {
                    "n_raw_hits": 2, "n_frames": 10,
                    "emit_rate": 0.0, "mean_emit_conf": None,
                },
            },
        },
    }
    html = html_clip(clip)
    assert "Filter to 1 camera" in html
    assert 'data-cam-filter="Cam4plus"' in html
    assert "overlay/5x5_clip_01_bestcam_t01285.1s_Cam4plus_boxes.mp4" in html
    assert "overlay/5x5_clip_01_bestcam_t01285.1s_Cam5plus_boxes.mp4" in html
    assert "Watching: Selected" in html
    assert "box-legend" in html
    assert "green EMIT" in html


def test_paint_view_marks_ball_after_resize():
    import numpy as np
    frame = np.zeros((2160, 3840, 3), dtype=np.uint8)
    pred = ([1920.0, 1080.0, 30.0, 30.0], 0.81, 30.0)
    vis = paint_view(frame, pred, pred, "P10", 1280)
    assert vis.shape[1] == 1280
    assert int(vis[:, :, 1].max()) > 200


def test_parse_clock_and_quad_windows():
    assert parse_clock("0:08") == 8.0
    assert parse_clock("6:52") == 412.0
    assert parse_clock("2:05") == 125.0
    with patch("run_5x5_ball_clips.video_duration_sec", return_value=1200.0):
        windows = named_quad_windows(MATCH2_CAMS)
    assert len(windows) == 4
    by_label = {w["label"]: w for w in windows}
    assert by_label["Center Start"]["start_sec"] == 8.0
    assert by_label["Center Start"]["duration_sec"] == 5.0
    assert by_label["Bottom Right"]["start_sec"] == 412.0
    assert by_label["Bottom Right"]["duration_sec"] == 6.0
    assert clip_stem(1, by_label["Top Left"]).startswith("quad_top_left_")


def test_quad_html_layout(tmp_path=None):
    from pathlib import Path
    import tempfile
    clip = {
        "stem": "quad_top_left_t00026.0s",
        "label": "Top Left",
        "slot": "top_left",
        "start_sec": 26.0,
        "duration_sec": 5.0,
        "cameras": [{"camera": "Cam4plus"}],
        "stats": {
            "n_frames": 10,
            "n_raw_hits": 10,
            "emit_rate": 0.0,
            "mean_selected_conf": 0.6,
            "top_camera": "Cam4plus",
            "overlay": "overlay/sel.mp4",
            "mosaic": "overlay/mos.mp4",
            "win_counts": {"Cam4plus": 10},
            "per_camera": {
                "Cam4plus": {
                    "n_raw_hits": 10, "n_frames": 10,
                    "emit_rate": 0.0, "mean_emit_conf": None,
                },
            },
        },
    }
    payload = {
        "title": "4 quad test",
        "layout": "quad",
        "seed": 1,
        "n_clips": 4,
        "clip_sec": 5.0,
        "min_thr": 0.3,
        "emit_thresh": 0.8,
        "cameras": "all_match2",
        "select_camera": "max_conf",
        "clips": [
            {**clip, "label": "Center Start", "slot": "center_start", "stem": "a"},
            {**clip, "label": "Bottom Right", "slot": "bottom_right", "stem": "b"},
            {**clip, "label": "Top Left", "slot": "top_left", "stem": "c"},
            {**clip, "label": "Top Right", "slot": "top_right", "stem": "d"},
        ],
    }
    labels = [c["label"] for c in ordered_clips(payload)]
    assert labels == ["Top Left", "Top Right", "Center Start", "Bottom Right"]
    out = Path(tempfile.mkdtemp())
    html = write_html(out, payload).read_text()
    assert "4 quad test" in html
    assert 'class="quad"' in html
    assert "grid-template-columns: 1fr 1fr" not in html
    assert html.index("Top Left") < html.index("Top Right")


def main() -> int:
    test_clip_stem_uses_5x5_name()
    test_clip_stem_bestcam()
    test_overlap_same_camera()
    test_pick_five_five_second_windows()
    test_pick_synced_windows_same_start()
    test_pick_selected_eight_cameras()
    test_pick_selected_all_empty()
    test_html_clip_has_per_camera_filter()
    test_paint_view_marks_ball_after_resize()
    test_parse_clock_and_quad_windows()
    test_quad_html_layout()
    print("ok")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
