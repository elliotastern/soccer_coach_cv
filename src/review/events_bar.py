"""Events bar overlay for coach review (Pass / Dribble / Movement / Recovery / Shot)."""
from __future__ import annotations

import cv2
import numpy as np

EVENT_TYPES = ("pass", "dribble", "movement", "recovery", "shot")
EVENT_BAR_H = 56
EVENT_COLORS_BGR = {
    "pass": (80, 180, 255),
    "shot": (60, 60, 255),
    "recovery": (80, 220, 120),
    "dribble": (200, 160, 80),
    "movement": (180, 180, 180),
}
EVENT_COLORS_PLOTLY = {
    "pass": "blue",
    "shot": "red",
    "dribble": "orange",
    "recovery": "purple",
    "movement": "gray",
}


def events_at_frame(events: list[dict], frame_id: int, window: int = 2) -> list[dict]:
    fid = int(frame_id)
    return [
        e
        for e in events
        if abs(int(e.get("start_frame", -10**9)) - fid) <= window
    ]


def events_up_to_frame(events: list[dict], frame_id: int) -> list[dict]:
    fid = int(frame_id)
    return [e for e in events if int(e.get("start_frame", 0)) <= fid]


def event_counts(events: list[dict]) -> dict[str, int]:
    counts = {t: 0 for t in EVENT_TYPES}
    for event in events:
        t = str(event.get("type", "unknown"))
        counts[t] = counts.get(t, 0) + 1
    return counts


def draw_events_bar(
    width: int,
    t_s: float,
    recent: list[dict],
    flash: str | None,
) -> np.ndarray:
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
        et = str(em.get("type", ""))
        label = f"{et} @{float(em.get('timestamp_end', em.get('t_end', 0))):.1f}s"
        col = EVENT_COLORS_BGR.get(et, (200, 200, 200))
        if flash and et == flash:
            col = tuple(min(255, c + 60) for c in col)
        cv2.rectangle(bar, (x, 28), (x + 118, 50), col, -1)
        cv2.putText(
            bar,
            label[:14],
            (x + 4, 45),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.38,
            (20, 20, 20),
            1,
        )
        x += 124
    if flash and not any(str(em.get("type")) == flash for em in recent[-3:]):
        col = EVENT_COLORS_BGR.get(flash, (255, 255, 255))
        cv2.putText(
            bar,
            f"NEW {flash.upper()}",
            (width - 180, 45),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            col,
            2,
        )
    return bar
