"""Plain-language coach UI helpers (non-technical mode)."""
from __future__ import annotations

SIMPLE_MODE_KEY = "coach_simple_mode"

BALL_VISIBLE_LABELS = {
    "yes": "Yes — I see the ball",
    "no": "No ball in view",
    "unclear": "Not sure",
}

QA_LABELS = {
    "good": "Looks good",
    "bad": "Something's wrong",
    "na": "Doesn't apply",
    "unset": "— pick one —",
}

GUIDE_STEPS = [
    "Use **Previous** / **Next** (or Play) to move through the match.",
    "Check the **video** (left) and **mini pitch map** (right): orange box on the ball, yellow dot on the map.",
    "Watch the **events bar** under the video — Pass, Dribble, Movement, Recovery, Shot.",
    "Answer the quick questions below, then click **Save this frame**.",
]


def is_simple_mode(session_state) -> bool:
    return bool(session_state.get(SIMPLE_MODE_KEY, True))


def ball_visible_options() -> tuple[str, ...]:
    return tuple(BALL_VISIBLE_LABELS.keys())


def qa_options() -> tuple[str, ...]:
    return ("good", "bad", "na", "unset")


def format_ball_visible(key: str) -> str:
    return BALL_VISIBLE_LABELS.get(key, key)


def format_qa(key: str) -> str:
    return QA_LABELS.get(key, key)


def qa_index(key: str) -> int:
    opts = qa_options()
    return opts.index(key) if key in opts else opts.index("unset")
