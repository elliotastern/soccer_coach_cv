#!/usr/bin/env python3
"""Unit tests for Pitch-1 H fit / calib write helpers."""
from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts" / "gold_set"))

from refit_match3_h_from_landmarks import (  # noqa: E402
    fit_h_from_paired_points,
    write_calib_h,
)


def test_fit_h_identity() -> None:
    pts = [[0.0, 0.0], [100.0, 0.0], [100.0, 50.0], [0.0, 50.0]]
    pitch = [[-10.0, -5.0], [10.0, -5.0], [10.0, 5.0], [-10.0, 5.0]]
    H = fit_h_from_paired_points(pts, pitch)
    assert H.shape == (3, 3)


def test_write_calib_h_preserves_meta() -> None:
    pts = [[0.0, 0.0], [100.0, 0.0], [100.0, 50.0], [0.0, 50.0]]
    pitch = [[-10.0, -5.0], [10.0, -5.0], [10.0, 5.0], [-10.0, 5.0]]
    H = fit_h_from_paired_points(pts, pitch)
    with tempfile.TemporaryDirectory() as td:
        path = Path(td) / "P10_manual.json"
        path.write_text(
            json.dumps(
                {
                    "H": np.eye(3).tolist(),
                    "image_points": pts,
                    "pitch_points": pitch,
                    "landmark_names": ["a", "b", "c", "d"],
                    "undistort": {"k1": -0.3, "alpha": 0.8},
                    "hull_image_points": [[1, 2], [3, 4]],
                }
            ),
            encoding="utf-8",
        )
        write_calib_h(path, H, backup=True, source="test")
        rec = json.loads(path.read_text(encoding="utf-8"))
        assert rec["undistort"]["k1"] == -0.3
        assert rec["hull_image_points"] == [[1, 2], [3, 4]]
        assert rec["h_source"] == "test"
        assert "H" in rec and "homography" in rec


if __name__ == "__main__":
    test_fit_h_identity()
    test_write_calib_h_preserves_meta()
    print("ok")
