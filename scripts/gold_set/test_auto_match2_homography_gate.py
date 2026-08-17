#!/usr/bin/env python3
"""Unit checks for Match 2 auto-H overlay gate (no model)."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts" / "gold_set"))

from auto_match2_homography import overlay_gate  # noqa: E402


def test_rejects_prior_bad_cam4():
    path = ROOT / "reports/eval_match2_v10/match2_pitch_calib/Cam4plus_top_left_auto.json"
    if not path.is_file():
        print("skip cam4 prior — missing json")
        return
    H = np.asarray(json.loads(path.read_text())["H"], dtype=float)
    gate = overlay_gate(H, (1920, 1080))
    assert gate["pass"] is False, gate
    assert "axis_aligned" in gate["reason"] or "span" in gate["reason"] or gate["reason"] != "ok"


def test_rejects_prior_bad_cam5():
    path = ROOT / "reports/eval_match2_v10/match2_pitch_calib/Cam5plus_top_left_auto.json"
    if not path.is_file():
        print("skip cam5 prior — missing json")
        return
    H = np.asarray(json.loads(path.read_text())["H"], dtype=float)
    gate = overlay_gate(H, (1920, 1080))
    assert gate["pass"] is False, gate


def test_plausible_perspective_passes():
    # Synthetic: image corners map to pitch corners with mild perspective
    img = np.float32([[200, 100], [1700, 120], [1800, 900], [150, 950]])
    pitch = np.float32([[-52.5, -34], [52.5, -34], [52.5, 34], [-52.5, 34]])
    import cv2

    H, _ = cv2.findHomography(img, pitch)
    gate = overlay_gate(H, (1920, 1080))
    assert gate["pass"] is True, gate


def test_rejects_thin_axis_aligned_band():
    # Synthetic full-width thin band (Cam5 failure mode)
    import cv2

    img = np.float32([[0, 400], [1280, 400], [1280, 520], [0, 520]])
    pitch = np.float32([[-52.5, -34], [52.5, -34], [52.5, 34], [-52.5, 34]])
    H, _ = cv2.findHomography(img, pitch)
    gate = overlay_gate(H, (1280, 720))
    assert gate["pass"] is False, gate
    assert "thin_axis_aligned_band" in gate["reason"] or "span_y" in gate["reason"]


if __name__ == "__main__":
    test_rejects_prior_bad_cam4()
    test_rejects_prior_bad_cam5()
    test_plausible_perspective_passes()
    test_rejects_thin_axis_aligned_band()
    print("ok")
