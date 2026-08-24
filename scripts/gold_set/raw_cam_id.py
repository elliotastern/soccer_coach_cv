"""Camera id is the P-code (or Goal id) in the video filename. Never remap by FOV."""
from __future__ import annotations

import re
from pathlib import Path

_P = re.compile(r"P(\d+)", re.I)


def cam_id_from_raw_name(name: str) -> str:
    stem = Path(name).stem
    low = stem.lower().replace(" ", "")
    if "goal1" in low:
        return "P_Goal1"
    if "goal2" in low:
        return "P_Goal2"
    if "4+" in stem.replace(" ", "") or "cam4+" in low:
        return "Cam4plus"
    if "5+" in stem.replace(" ", "") or "cam5+" in low:
        return "Cam5plus"
    found = _P.findall(stem)
    if not found:
        raise ValueError(f"no camera id in filename: {name}")
    return f"P{int(found[-1])}"


def load_match_raw(match_dir: Path | str) -> dict[str, Path]:
    match_dir = Path(match_dir)
    out: dict[str, Path] = {}
    if not match_dir.is_dir():
        raise FileNotFoundError(match_dir)
    for path in sorted(match_dir.glob("*.mp4")):
        if path.name.startswith("._"):
            continue
        try:
            cam = cam_id_from_raw_name(path.name)
        except ValueError:
            # Recorder originals (e.g. cam-N_...) beside P-code symlinks — skip.
            continue
        if cam in out:
            raise ValueError(f"duplicate {cam}: {out[cam].name} and {path.name}")
        out[cam] = path
    return out
