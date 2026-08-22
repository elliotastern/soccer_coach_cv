"""Persist per-frame human labels for Phase 1 review (growth engineering)."""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

LABEL_FILE = "labels.json"
BALL_VISIBLE = ("yes", "no", "unclear")
QA_OK = ("good", "bad", "na", "unset")


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def labels_path(run_dir: Path) -> Path:
    return Path(run_dir) / LABEL_FILE


def empty_labels() -> Dict[str, Any]:
    return {"updated_at": _now(), "reviewer": "", "frames": {}}


def load_labels(run_dir: Path) -> Dict[str, Any]:
    path = labels_path(run_dir)
    if not path.is_file():
        return empty_labels()
    data = json.loads(path.read_text(encoding="utf-8"))
    data.setdefault("frames", {})
    return data


def save_labels(run_dir: Path, data: Dict[str, Any]) -> Path:
    data["updated_at"] = _now()
    path = labels_path(run_dir)
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")
    return path


def frame_key(frame_id: int) -> str:
    return str(int(frame_id))


def get_frame_label(data: Dict[str, Any], frame_id: int) -> Dict[str, Any]:
    raw = data.get("frames", {}).get(frame_key(frame_id), {})
    return {
        "ball_visible": raw.get("ball_visible", "unclear"),
        "ball_box_ok": raw.get("ball_box_ok", "unset"),
        "pitch_ball_ok": raw.get("pitch_ball_ok", "unset"),
        "team_ok": raw.get("team_ok", "unset"),
        "event_ok": raw.get("event_ok", "unset"),
        "flag": bool(raw.get("flag", False)),
        "note": str(raw.get("note", "")),
        "reviewed": bool(raw.get("reviewed", False)),
    }


def set_frame_label(data: Dict[str, Any], frame_id: int, fields: Dict[str, Any]) -> None:
    key = frame_key(frame_id)
    cur = dict(data.setdefault("frames", {}).get(key, {}))
    for k, v in fields.items():
        cur[k] = v
    cur["reviewed"] = True
    cur["reviewed_at"] = _now()
    data["frames"][key] = cur


def labeled_frames(data: Dict[str, Any]) -> List[int]:
    out = []
    for k, v in data.get("frames", {}).items():
        if v.get("reviewed"):
            out.append(int(k))
    return sorted(out)


def flagged_frames(data: Dict[str, Any]) -> List[int]:
    out = []
    for k, v in data.get("frames", {}).items():
        if v.get("flag"):
            out.append(int(k))
    return sorted(out)


def label_stats(data: Dict[str, Any]) -> Dict[str, int]:
    frames = data.get("frames", {})
    reviewed = sum(1 for v in frames.values() if v.get("reviewed"))
    flagged = sum(1 for v in frames.values() if v.get("flag"))
    bad_ball = sum(
        1
        for v in frames.values()
        if v.get("ball_box_ok") == "bad" or v.get("pitch_ball_ok") == "bad"
    )
    return {"reviewed": reviewed, "flagged": flagged, "bad_ball": bad_ball, "total": len(frames)}


def low_conf_event_frames(events: List[Dict], max_conf: float = 0.80) -> List[int]:
    out = []
    for e in events:
        conf = float(e.get("confidence", 0.0))
        if conf < max_conf:
            out.append(int(e.get("start_frame", 0)))
    return sorted(set(out))
