#!/usr/bin/env python3
"""Re-prelabel a frame range in gold100 XML; preserve human-corrected frames."""
from __future__ import annotations

import argparse
import shutil
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import cv2

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.perception.rfdetr_local import LocalRFDETRDetector


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--gold-dir", type=Path, default=ROOT / "data/processed/gold_sets/match1_1_100")
    p.add_argument("--keep-through", type=int, default=20, help="Keep frames 0..N inclusive")
    p.add_argument("--start", type=int, default=21)
    p.add_argument("--end", type=int, default=99, help="Inclusive strip frame index")
    p.add_argument("--player-checkpoint", type=Path, default=ROOT / "models/people_after_100_epochs.pth")
    p.add_argument("--ball-checkpoint", type=Path, default=ROOT / "models/ball_89.pth")
    p.add_argument("--player-thr", type=float, default=0.5)
    p.add_argument("--enhance-ball", action="store_true", default=True)
    p.add_argument("--no-enhance-ball", action="store_true")
    return p.parse_args()


def backup_xml(xml_path: Path) -> Path:
    bak = xml_path.with_suffix(".xml.bak_before_reprelabel")
    shutil.copy2(xml_path, bak)
    return bak


def strip_frames_from_xml(root: ET.Element, start: int, end: int) -> int:
    removed = 0
    for track in list(root.findall("track")):
        label = (track.get("label") or "").lower()
        if label not in ("player", "person", "ball"):
            continue
        for box in list(track.findall("box")):
            f = int(box.get("frame"))
            if start <= f <= end:
                track.remove(box)
                removed += 1
        if not track.findall("box"):
            root.remove(track)
    return removed


def next_track_id(root: ET.Element) -> int:
    ids = []
    for track in root.findall("track"):
        try:
            ids.append(int(track.get("id")))
        except (TypeError, ValueError):
            continue
    return (max(ids) + 1) if ids else 0


def add_detection_track(root: ET.Element, track_id: int, frame: int, det, source: str = "auto") -> None:
    label = det.class_name if det.class_name != "person" else "player"
    track = ET.SubElement(root, "track", {
        "id": str(track_id),
        "label": label,
        "source": source,
    })
    x, y, w, h = det.bbox
    box = ET.SubElement(track, "box", {
        "frame": str(frame),
        "xtl": f"{x:.2f}",
        "ytl": f"{y:.2f}",
        "xbr": f"{x + w:.2f}",
        "ybr": f"{y + h:.2f}",
        "outside": "0",
        "occluded": "0",
        "keyframe": "1",
    })
    attr = ET.SubElement(box, "attribute", {"name": "confidence"})
    attr.text = f"{det.confidence:.3f}"


def count_boxes(root: ET.Element, lo: int, hi: int) -> tuple[int, int]:
    n_player = n_ball = 0
    for track in root.findall("track"):
        label = (track.get("label") or "").lower()
        for box in track.findall("box"):
            f = int(box.get("frame"))
            if not (lo <= f <= hi):
                continue
            if label == "ball":
                n_ball += 1
            elif label in ("player", "person"):
                n_player += 1
    return n_player, n_ball


def main():
    args = parse_args()
    gold = args.gold_dir.resolve()
    xml_path = gold / "prelabels" / "annotations.xml"
    if not xml_path.is_file():
        raise FileNotFoundError(xml_path)

    start, end = args.start, args.end
    keep_through = args.keep_through
    if start <= keep_through:
        raise ValueError(f"--start {start} must be > keep-through {keep_through}")

    bak = backup_xml(xml_path)
    print(f"Backup: {bak}", flush=True)

    tree = ET.parse(xml_path)
    root = tree.getroot()
    kept_p, kept_b = count_boxes(root, 0, keep_through)
    print(f"Preserve frames 0-{keep_through}: player={kept_p} ball={kept_b}", flush=True)

    removed = strip_frames_from_xml(root, start, end)
    print(f"Removed {removed} boxes on frames {start}-{end}", flush=True)

    enhance = args.enhance_ball and not args.no_enhance_ball
    print(f"Loading detector enhance_ball={enhance}...", flush=True)
    det = LocalRFDETRDetector(
        player_checkpoint=str(args.player_checkpoint),
        ball_checkpoint=str(args.ball_checkpoint),
        confidence_threshold=args.player_thr,
        enhance_ball=enhance,
    )

    tid = next_track_id(root)
    added = 0
    for frame in range(start, end + 1):
        path = gold / "review" / "frames" / f"{frame:03d}.jpg"
        img = cv2.imread(str(path))
        if img is None:
            raise RuntimeError(f"Missing review frame: {path}")
        outs = det.detect(img)
        for d in outs:
            if d.class_name not in ("player", "person", "ball"):
                continue
            add_detection_track(root, tid, frame, d, source="auto")
            tid += 1
            added += 1
        n_p = sum(1 for d in outs if d.class_name in ("player", "person"))
        n_b = sum(1 for d in outs if d.class_name == "ball")
        print(f"frame {frame:03d}: +{n_p} player +{n_b} ball", flush=True)

    # Preserve frames 0-keep unchanged check
    after_p, after_b = count_boxes(root, 0, keep_through)
    if (after_p, after_b) != (kept_p, kept_b):
        raise RuntimeError(
            f"Preserve check failed: before player/ball=({kept_p},{kept_b}) "
            f"after=({after_p},{after_b})"
        )

    new_p, new_b = count_boxes(root, start, end)
    tree.write(xml_path, encoding="utf-8", xml_declaration=True)
    print(
        f"Wrote {xml_path}\n"
        f"Frames {start}-{end}: player={new_p} ball={new_b} (added tracks/boxes={added})\n"
        f"Frames 0-{keep_through} unchanged: player={after_p} ball={after_b}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
