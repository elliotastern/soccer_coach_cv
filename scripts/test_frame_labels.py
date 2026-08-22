#!/usr/bin/env python3
"""Smoke test frame_labels persistence."""
from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.review.frame_labels import (  # noqa: E402
    flagged_frames,
    get_frame_label,
    label_stats,
    load_labels,
    low_conf_event_frames,
    save_labels,
    set_frame_label,
)


def main() -> int:
    with tempfile.TemporaryDirectory() as tmp:
        run = Path(tmp)
        data = load_labels(run)
        set_frame_label(
            data,
            2400,
            {
                "ball_visible": "yes",
                "ball_box_ok": "good",
                "pitch_ball_ok": "good",
                "team_ok": "good",
                "event_ok": "na",
                "flag": True,
                "note": "test",
            },
        )
        save_labels(run, data)
        path = run / "labels.json"
        assert path.is_file()
        reloaded = load_labels(run)
        cur = get_frame_label(reloaded, 2400)
        assert cur["ball_visible"] == "yes"
        assert cur["flag"] is True
        assert flagged_frames(reloaded) == [2400]
        stats = label_stats(reloaded)
        assert stats["reviewed"] == 1
        assert stats["flagged"] == 1
        low = low_conf_event_frames(
            [{"start_frame": 100, "confidence": 0.5}, {"start_frame": 200, "confidence": 0.9}]
        )
        assert low == [100]
        print("OK", json.dumps(stats))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
