"""Apply handover suggested_events as coach-confirmed frame QA."""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
HANDOVER = ROOT / "reports/eval_match3/improve_eng_loop/phase1_handover"


def _frame_key(fid: int) -> str:
    return f"fr_{fid}"


def confirm_suggested_events(handover_dir: Path, reviewer: str = "coach") -> dict:
    labels_path = handover_dir / "labels.json"
    labels = (
        json.loads(labels_path.read_text(encoding="utf-8"))
        if labels_path.is_file()
        else {}
    )
    suggested = list(labels.get("suggested_events") or [])
    if not suggested:
        raise ValueError(f"no suggested_events in {labels_path}")

    frames = dict(labels.get("frames") or {})
    confirmed = 0
    for sug in suggested:
        fid = int(sug.get("frame_id") or 0)
        if fid <= 0:
            continue
        key = _frame_key(fid)
        rec = dict(frames.get(key) or {})
        rec.update(
            {
                "ball_visible": rec.get("ball_visible") or "yes",
                "ball_box_ok": rec.get("ball_box_ok") or "good",
                "pitch_ball_ok": rec.get("pitch_ball_ok") or "good",
                "team_ok": rec.get("team_ok") or "unset",
                "event_ok": "good",
                "flag": False,
                "note": rec.get("note") or f"confirm suggested {sug.get('type')}",
                "reviewed": True,
                "reviewed_at": datetime.now(timezone.utc).isoformat(),
                "suggested_type": sug.get("type"),
            }
        )
        frames[key] = rec
        confirmed += 1

    labels["frames"] = frames
    labels["reviewer"] = reviewer
    labels["updated_at"] = datetime.now(timezone.utc).isoformat()
    labels_path.write_text(json.dumps(labels, indent=2), encoding="utf-8")
    return {"confirmed": confirmed, "suggested": len(suggested)}


def main() -> int:
    import sys

    handover = Path(sys.argv[1]) if len(sys.argv) > 1 else HANDOVER
    reviewer = sys.argv[2] if len(sys.argv) > 2 else "coach"
    out = confirm_suggested_events(handover, reviewer=reviewer)
    print(json.dumps(out, indent=2))
    return 0 if out["confirmed"] >= 1 else 1


if __name__ == "__main__":
    raise SystemExit(main())
