#!/usr/bin/env python3
"""Unit checks for build_4quad_cvat_pack."""
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from scripts.gold_set.build_4quad_cvat_pack import CLIPS, CAMERA, OUT_DEFAULT, source_path


def test_four_clips():
    assert len(CLIPS) == 4
    assert CAMERA == "Cam4plus"


def test_sources_exist():
    for clip in CLIPS:
        path = source_path(clip["stem"], clip["camera"])
        assert path.is_file(), path


def test_per_window_cameras():
    cams = {c["slot"]: c["camera"] for c in CLIPS}
    assert cams["top_left"] == "P10"
    assert cams["top_right"] == "P7"
    assert cams["bottom_right"] == "Cam5plus"
    assert cams["center_start"] == "Cam4plus"


def test_editor_not_truncated():
    editor = OUT_DEFAULT / "review" / "editor.html"
    if not editor.is_file():
        return
    html = editor.read_text(encoding="utf-8")
    assert "loadReviewFrame" in html
    assert "match2_4quad_label/gold/annotations.xml" in html
    assert editor.stat().st_size > 50000


def test_gold_uses_tracks():
    gold = OUT_DEFAULT / "gold" / "annotations.xml"
    if not gold.is_file():
        return
    text = gold.read_text(encoding="utf-8")
    assert "<track " in text
    assert text.count("<track ") > 0


if __name__ == "__main__":
    test_four_clips()
    test_sources_exist()
    test_per_window_cameras()
    test_editor_not_truncated()
    test_gold_uses_tracks()
    print("ok")
