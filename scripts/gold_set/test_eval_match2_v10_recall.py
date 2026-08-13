#!/usr/bin/env python3
"""Unit checks for no-emit-gate recall scoring."""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts" / "gold_set"))

from eval_match2_v10_recall import score


def test_score_recall_and_frame_hit():
    items = [
        {"camera": "Cam4plus", "image": "a.jpg", "gt": [[10.0, 10.0, 20.0, 20.0]]},
        {"camera": "P10", "image": "b.jpg", "gt": [[10.0, 10.0, 20.0, 20.0]]},
    ]
    preds = {
        "a.jpg": [([11.0, 11.0, 20.0, 20.0], 0.91, 20.0)],
        "b.jpg": [([200.0, 200.0, 20.0, 20.0], 0.91, 20.0)],
    }
    block = score(items, preds, 0.5)
    assert block["tp"] == 1
    assert block["fp"] == 1
    assert block["fn"] == 1
    assert abs(block["recall"] - 0.5) < 1e-9
    assert block["frames_with_hit"] == 1
    assert abs(block["by_camera"]["Cam4plus"]["recall"] - 1.0) < 1e-9
    assert abs(block["by_camera"]["P10"]["recall"] - 0.0) < 1e-9


def main() -> int:
    test_score_recall_and_frame_hit()
    print("ok")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
