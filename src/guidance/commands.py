"""Haptic guidance command encoding (Product Phase 2+).

Wrist action signals and shin 360-degree PWM direction codes.
"""

# Reserved hex range for directional guidance (Phase 2+)
DIRECTION_HEX_MIN = 0x10
DIRECTION_HEX_MAX = 0xFF


def encode_shin_direction(angle_deg: float) -> int:
    """Map 0-360 degrees to a guidance hex code in [0x10, 0xFF]."""
    angle = angle_deg % 360.0
    span = DIRECTION_HEX_MAX - DIRECTION_HEX_MIN
    code = DIRECTION_HEX_MIN + int(round((angle / 360.0) * span))
    return min(max(code, DIRECTION_HEX_MIN), DIRECTION_HEX_MAX)


def encode_wrist_action(action: str) -> str:
    """Map a discrete action name to a wrist cue label (Phase 2+ protocol TBD)."""
    allowed = {"pass_ground", "pass_lofted", "shoot", "dribble", "recover", "none"}
    key = action.strip().lower()
    if key not in allowed:
        raise ValueError(f"Unknown wrist action: {action}")
    return key
