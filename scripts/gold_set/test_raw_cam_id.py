#!/usr/bin/env python3
"""Camera id is the P-code in the video filename."""
from __future__ import annotations

from raw_cam_id import cam_id_from_raw_name


def main() -> int:
    cases = {
        "P1-006.mp4": "P1",
        "P9-004.mp4": "P9",
        "P10-002.mp4": "P10",
        "p8-005.mp4": "P8",
        "P_Goal1-007.mp4": "P_Goal1",
        "P_Goal2-008.mp4": "P_Goal2",
        "Cam 3-P1.mp4": "P1",
        "Cam 8-P10-003.mp4": "P10",
        "Cam 10-P12-001.mp4": "P12",
        "Cam 4+-002.mp4": "Cam4plus",
        "Cam 5+-004.mp4": "Cam5plus",
    }
    for name, want in cases.items():
        got = cam_id_from_raw_name(name)
        if got != want:
            print(f"{name} -> {got} want {want}")
            return 1
    if cam_id_from_raw_name("P1-006.mp4") == "P9":
        print("P1-006.mp4 must not be P9")
        return 1
    print("raw cam ids ok")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
