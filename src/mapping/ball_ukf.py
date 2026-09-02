"""Constant-velocity UKF on pitch (x, y) for 3D-fused ball measurements."""
from __future__ import annotations

import numpy as np

from src.mapping.match3_xy import EMIT_CONF, HOLD_MAX_GAP

DT_DEFAULT = 1.0 / 60.0
COAST_DECAY = 0.92


class BallPitchUKF:
    """State [x, y, vx, vy]; predict/update/coast for fused ball ticks."""

    def __init__(self, dt: float = DT_DEFAULT, emit_conf: float = EMIT_CONF):
        self.dt = float(dt)
        self.emit_conf = float(emit_conf)
        self.x = np.zeros(4, dtype=float)
        self.P = np.eye(4, dtype=float) * 10.0
        self.ready = False
        self.coast_age = 0
        self.last_conf = 0.0

    def predict(self) -> tuple[float, float]:
        F = np.array(
            [[1, 0, self.dt, 0], [0, 1, 0, self.dt], [0, 0, 1, 0], [0, 0, 0, 1]],
            dtype=float,
        )
        q = 0.05
        Q = np.array(
            [
                [q, 0, 0, 0],
                [0, q, 0, 0],
                [0, 0, q * 4, 0],
                [0, 0, 0, q * 4],
            ],
            dtype=float,
        )
        self.x = F @ self.x
        self.P = F @ self.P @ F.T + Q
        return float(self.x[0]), float(self.x[1])

    def update(self, xy, conf: float) -> tuple[float, float]:
        z = np.array([float(xy[0]), float(xy[1])], dtype=float)
        H = np.array([[1, 0, 0, 0], [0, 1, 0, 0]], dtype=float)
        r = max(0.05, (1.0 - min(float(conf), 0.99)) * 2.0)
        R = np.eye(2, dtype=float) * r
        if not self.ready:
            self.x[0], self.x[1] = z[0], z[1]
            self.x[2] = self.x[3] = 0.0
            self.ready = True
            self.coast_age = 0
            self.last_conf = float(conf)
            return float(self.x[0]), float(self.x[1])
        y = z - H @ self.x
        S = H @ self.P @ H.T + R
        K = self.P @ H.T @ np.linalg.inv(S)
        self.x = self.x + K @ y
        self.P = (np.eye(4) - K @ H) @ self.P
        self.coast_age = 0
        self.last_conf = float(conf)
        return float(self.x[0]), float(self.x[1])

    def coast_conf(self, base_conf: float) -> float:
        return float(self.last_conf or base_conf) * (COAST_DECAY ** self.coast_age)

    def step(
        self,
        fused: dict | None,
        *,
        hold_max_gap: int = HOLD_MAX_GAP,
    ) -> dict | None:
        """Predict; update on measurement; else coast if gap small enough."""
        self.predict()
        if fused is not None and float(fused.get("conf", 0.0)) >= self.emit_conf:
            xy = self.update(fused["xy"], float(fused["conf"]))
            out = dict(fused)
            out["xy"] = xy
            out["ukf"] = True
            return out
        self.coast_age += 1
        if not self.ready or self.coast_age > int(hold_max_gap):
            return None
        conf = self.coast_conf(self.emit_conf)
        if conf < self.emit_conf:
            return None
        return {
            "xy": (float(self.x[0]), float(self.x[1])),
            "z": 0.0,
            "conf": conf,
            "cam": "ukf_coast",
            "n": 0,
            "agree": False,
            "reproj_inliers": 0,
            "fuse_mode": "triangulate_3d",
            "ukf": True,
            "coast": True,
        }
