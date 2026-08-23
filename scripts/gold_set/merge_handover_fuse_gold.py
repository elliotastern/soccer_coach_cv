"""Merge coach handover frame QA into fuse event gold windows."""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
HANDOVER = ROOT / "reports/eval_match3/improve_eng_loop/phase1_handover"
CLIP = ROOT / "data/processed/gold_sets/match3_events_v2_dribble/clips/real_fuse_15s"


def _frame_key(fid: int) -> str:
    return f"fr_{fid}"


def _load_emits(handover: Path) -> list[dict]:
    for name in ("emits_render.json", "meta.json"):
        p = handover / name
        if not p.is_file():
            continue
        data = json.loads(p.read_text(encoding="utf-8"))
        if name == "meta.json":
            return list(data.get("emits") or [])
        return list(data.get("emits") or [])
    return []


def merge_handover_labels(handover_dir: Path, base_labels: dict) -> dict:
    labels_path = handover_dir / "labels.json"
    handover = json.loads(labels_path.read_text(encoding="utf-8")) if labels_path.is_file() else {}
    frames = handover.get("frames") or {}
    emits = _load_emits(handover_dir)
    if not emits:
        return dict(base_labels)

    coach_events: list[dict] = []
    coach_negs: list[dict] = []
    for em in emits:
        fid = int(em.get("frame_id") or 0)
        key = _frame_key(fid)
        fr = frames.get(key) or frames.get(str(fid)) or {}
        ok = fr.get("event_ok")
        if ok == "good":
            coach_events.append(
                {
                    "type": em["type"],
                    "t_start": float(em.get("t_start", em.get("t_end", 0))),
                    "t_end": float(em.get("t_end", 0)),
                    "note": f"coach confirm fr {fid}",
                    "source": "handover",
                }
            )
        elif ok == "bad":
            t = float(em.get("t_end", 0))
            coach_negs.append(
                {
                    "t_start": max(0.0, t - 0.35),
                    "t_end": t + 0.35,
                    "note": f"coach reject {em['type']} fr {fid}",
                    "source": "handover",
                }
            )

    out = dict(base_labels)
    if coach_events:
        seen = {(e["type"], e["t_start"], e["t_end"]) for e in base_labels.get("events") or []}
        merged = list(base_labels.get("events") or [])
        for e in coach_events:
            key = (e["type"], e["t_start"], e["t_end"])
            if key not in seen:
                merged.append(e)
                seen.add(key)
        out["events"] = merged
        out["coach_confirmed"] = len(coach_events)
    if coach_negs:
        negs = list(base_labels.get("negatives") or [])
        negs.extend(coach_negs)
        out["negatives"] = negs
    out["handover_reviewer"] = handover.get("reviewer") or ""
    out["handover_updated_at"] = handover.get("updated_at") or ""
    return out


def seed_handover_suggestions(handover_dir: Path) -> None:
    """Add detector emits as suggested_events when labels.json is empty."""
    labels_path = handover_dir / "labels.json"
    labels = json.loads(labels_path.read_text(encoding="utf-8")) if labels_path.is_file() else {}
    emits = _load_emits(handover_dir)
    if emits and not labels.get("suggested_events"):
        labels["suggested_events"] = [
            {
                "type": e["type"],
                "t_start": e.get("t_start"),
                "t_end": e.get("t_end"),
                "confidence": e.get("confidence"),
                "frame_id": e.get("frame_id"),
            }
            for e in emits
        ]
        labels_path.write_text(json.dumps(labels, indent=2), encoding="utf-8")


def main() -> int:
    CLIP.mkdir(parents=True, exist_ok=True)
    base_path = CLIP / "labels.json"
    base = json.loads(base_path.read_text(encoding="utf-8")) if base_path.is_file() else {"events": []}
    if HANDOVER.is_dir():
        seed_handover_suggestions(HANDOVER)
    merged = merge_handover_labels(HANDOVER, base)
    out_path = CLIP / "labels_merged.json"
    out_path.write_text(json.dumps(merged, indent=2), encoding="utf-8")
    base_path.write_text(json.dumps(merged, indent=2), encoding="utf-8")
    print("WROTE", out_path, "events=", len(merged.get("events") or []))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
