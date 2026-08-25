#!/usr/bin/env python3
"""Build multicam fuse eval clip + labels (separate from handover window)."""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
V2_MANIFEST = ROOT / "data/processed/gold_sets/match3_events_v2_dribble/manifest.json"
BASE = ROOT / "data/processed/gold_sets/match3_events_v2_dribble/clips"


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--start", type=int, required=True)
    p.add_argument("--match-sec", type=float, default=15.0)
    p.add_argument("--stride", type=int, default=4)
    p.add_argument("--clip-id", type=str, required=True)
    p.add_argument("--labels-json", type=Path, help="Optional labels file to copy")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    out = BASE / args.clip_id
    out.mkdir(parents=True, exist_ok=True)
    cmd = [
        sys.executable,
        str(ROOT / "scripts/gold_set/build_check25_event_timeline.py"),
        "--start",
        str(args.start),
        "--match-sec",
        str(args.match_sec),
        "--stride",
        str(args.stride),
        "--out-dir",
        str(out),
    ]
    rc = subprocess.call(cmd, cwd=str(ROOT))
    if rc != 0:
        return rc
    if args.labels_json and args.labels_json.is_file():
        labels = json.loads(args.labels_json.read_text(encoding="utf-8"))
    else:
        labels = {
            "source": f"eval window start={args.start}",
            "start_frame": args.start,
            "stride": args.stride,
            "match_sec": args.match_sec,
            "holdout": True,
            "events": [],
            "negatives": [],
        }
    labels["start_frame"] = args.start
    labels["stride"] = args.stride
    labels["match_sec"] = args.match_sec
    (out / "labels.json").write_text(json.dumps(labels, indent=2), encoding="utf-8")

    entry = {
        "id": args.clip_id,
        "timeline": str((out / "timeline.json").relative_to(ROOT)),
        "labels": str((out / "labels.json").relative_to(ROOT)),
        "note": f"Eval fuse window start={args.start}s span={args.match_sec}",
        "holdout": True,
    }
    if V2_MANIFEST.is_file():
        manifest = json.loads(V2_MANIFEST.read_text(encoding="utf-8"))
        clips = [c for c in manifest.get("clips") or [] if c.get("id") != args.clip_id]
        clips.append(entry)
        manifest["clips"] = clips
        V2_MANIFEST.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    else:
        print("SKIP manifest update (missing", V2_MANIFEST, ")", flush=True)
    print("WROTE", out / "labels.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
