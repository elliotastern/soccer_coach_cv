#!/usr/bin/env python3
"""Pick winner on train metrics only."""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts" / "gold_set"))

from tune_track_enhancements import pick_winner


def test_picks_higher_recall_if_precise():
    rows = [
        {"name": "detector_only", "train": {"recall": 0.90, "precision": 0.97}},
        {"name": "bt", "train": {"recall": 0.93, "precision": 0.95}},
        {"name": "flood", "train": {"recall": 0.99, "precision": 0.40}},
    ]
    win = pick_winner(rows, 0.97)
    assert win["name"] == "bt"


def test_keeps_detector_if_tracker_worse():
    rows = [
        {"name": "detector_only", "train": {"recall": 0.92, "precision": 0.96}},
        {"name": "legacy", "train": {"recall": 0.71, "precision": 0.88}},
    ]
    win = pick_winner(rows, 0.96)
    assert win["name"] == "detector_only"


def main() -> int:
    test_picks_higher_recall_if_precise()
    test_keeps_detector_if_tracker_worse()
    print("ok")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
