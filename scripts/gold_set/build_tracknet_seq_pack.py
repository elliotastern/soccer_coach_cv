#!/usr/bin/env python3
"""Build a TrackNet-style multi-frame pack from continuous gold.

Sources (validated continuous only):
  - match2_4quad_top_left (P10)
  - match2_4quad_top_left_p7 (P7)
  - match2_4quad_top_left_cam4plus (Cam4plus)
  - match3_quad_p10_31 (P10, human-reviewed)

Holdouts match v11 temporal policy:
  train mid-frame 1–219 (4quad) / 1–99 (Match3)
  valid mid-frame 220–239 / 100–119
  test  mid-frame 240–end / 120–193

Does not include Match3 P8 (not human-reviewed) or sparse packs.
"""
from __future__ import annotations

import argparse
import json
import shutil
import xml.etree.ElementTree as ET
from collections import defaultdict
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
GOLD = ROOT / "data/processed/gold_sets"
OUT_DEFAULT = Path("/Volumes/LaCie/Projects/Soccer project data/ball_tracknet_seq_v1")

# Heatmap size used by many TrackNet forks (W×H).
HM_W = 512
HM_H = 288
SIGMA = 2.5

QUAD_CLIPS = [
    {
        "id": "match2_4quad_top_left",
        "camera": "P10",
        "match": "match2",
        "clock": "0:26-0:31",
        "n_frames": 300,
        "label_kind": "cvat_track_xml",
    },
    {
        "id": "match2_4quad_top_left_p7",
        "camera": "P7",
        "match": "match2",
        "clock": "0:26-0:31",
        "n_frames": 300,
        "label_kind": "cvat_track_xml",
    },
    {
        "id": "match2_4quad_top_left_cam4plus",
        "camera": "Cam4plus",
        "match": "match2",
        "clock": "0:26-0:31",
        "n_frames": 299,
        "label_kind": "cvat_track_xml",
    },
]

MATCH3_CLIP = {
    "id": "match3_quad_p10_31",
    "camera": "P10",
    "match": "match3",
    "clock": "0:31-0:36",
    "n_frames": 299,
    "label_kind": "match3_labels_json",
    "focus_cam": "P10",
}


def frame_path(frames_dir: Path, i: int) -> Path | None:
    for name in (f"{i:03d}.jpg", f"{i:04d}.jpg"):
        p = frames_dir / name
        if p.is_file():
            return p
    return None


def xml_ball_boxes(xml_path: Path) -> dict[int, list]:
    root = ET.parse(xml_path).getroot()
    raw = defaultdict(list)
    for track in root.findall("track"):
        if (track.get("label") or "").lower() != "ball":
            continue
        for box in track.findall("box"):
            if box.get("outside") == "1":
                continue
            frame = int(box.get("frame"))
            xtl = float(box.get("xtl"))
            ytl = float(box.get("ytl"))
            xbr = float(box.get("xbr"))
            ybr = float(box.get("ybr"))
            raw[frame].append([xtl, ytl, xbr - xtl, ybr - ytl])
    return dict(raw)


def match3_ball_boxes(lab_path: Path, focus_cam: str) -> dict[int, list]:
    lab = json.loads(lab_path.read_text())
    raw = {}
    for fr in lab["frames"]:
        balls = fr["cams"][focus_cam].get("gt_balls") or []
        if not balls:
            continue
        boxes = []
        for bb in balls:
            boxes.append([float(bb["x"]), float(bb["y"]), float(bb["w"]), float(bb["h"])])
        raw[int(fr["i"])] = boxes
    return raw


def pick_primary_box(boxes: list) -> list | None:
    if not boxes:
        return None
    return max(boxes, key=lambda b: b[2] * b[3])


def box_center(box: list) -> tuple[float, float]:
    x, y, w, h = box
    return x + w / 2.0, y + h / 2.0


def split_for_mid(mid: int, train_hi: int, valid_lo: int, valid_hi: int, test_lo: int) -> str | None:
    if 1 <= mid <= train_hi:
        return "train"
    if valid_lo <= mid <= valid_hi:
        return "valid"
    if mid >= test_lo:
        return "test"
    return None


def gaussian_heatmap(cx: float, cy: float, fw: int, fh: int) -> np.ndarray:
    """cx,cy in full image pixels → HM_W×HM_H uint8 peak map."""
    sx = cx * (HM_W / float(fw))
    sy = cy * (HM_H / float(fh))
    xs = np.arange(HM_W, dtype=np.float32)
    ys = np.arange(HM_H, dtype=np.float32)
    xx, yy = np.meshgrid(xs, ys)
    heat = np.exp(-((xx - sx) ** 2 + (yy - sy) ** 2) / (2.0 * SIGMA**2))
    return np.clip(heat * 255.0, 0, 255).astype(np.uint8)


def link_or_copy(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() or dst.is_symlink():
        dst.unlink()
    try:
        dst.symlink_to(src.resolve())
    except OSError:
        shutil.copy2(src, dst)


def load_clip(meta: dict) -> dict:
    pack = GOLD / meta["id"]
    frames_dir = pack / "review" / "frames"
    if not frames_dir.is_dir():
        raise FileNotFoundError(f"missing frames: {frames_dir}")
    if meta["label_kind"] == "cvat_track_xml":
        boxes = xml_ball_boxes(pack / "gold" / "annotations.xml")
    else:
        boxes = match3_ball_boxes(pack / "labels.json", meta["focus_cam"])
    present = []
    for i in range(meta["n_frames"]):
        p = frame_path(frames_dir, i)
        if p is None:
            raise FileNotFoundError(f"{meta['id']} missing frame {i}")
        present.append(p)
    # image size from first frame
    with Image.open(present[0]) as im:
        w, h = im.size
    return {
        "meta": meta,
        "frames": present,
        "boxes": boxes,
        "width": w,
        "height": h,
        "src_pack": pack,
    }


def write_clip_labels(out_clip: Path, clip: dict) -> None:
    rows = []
    for i, src in enumerate(clip["frames"]):
        box = pick_primary_box(clip["boxes"].get(i, []))
        visible = 1 if box is not None else 0
        cx = cy = None
        bbox = None
        if box is not None:
            cx, cy = box_center(box)
            bbox = [round(box[0], 2), round(box[1], 2), round(box[2], 2), round(box[3], 2)]
        rows.append(
            {
                "frame": i,
                "file": f"{i:04d}.jpg",
                "visible": visible,
                "cx": None if cx is None else round(cx, 2),
                "cy": None if cy is None else round(cy, 2),
                "bbox_xywh": bbox,
            }
        )
        link_or_copy(src, out_clip / "frames" / f"{i:04d}.jpg")
    (out_clip / "labels.json").write_text(
        json.dumps(
            {
                "clip_id": clip["meta"]["id"],
                "camera": clip["meta"]["camera"],
                "match": clip["meta"]["match"],
                "clock": clip["meta"]["clock"],
                "width": clip["width"],
                "height": clip["height"],
                "n_frames": len(rows),
                "n_visible": sum(r["visible"] for r in rows),
                "frames": rows,
            },
            indent=2,
        ),
        encoding="utf-8",
    )


def build_triplets(clip: dict, split_cfg: dict) -> list[dict]:
    meta = clip["meta"]
    train_hi = split_cfg["train_hi"]
    valid_lo = split_cfg["valid_lo"]
    valid_hi = split_cfg["valid_hi"]
    test_lo = split_cfg["test_lo"]
    n = len(clip["frames"])
    out = []
    for mid in range(1, n - 1):
        split = split_for_mid(mid, train_hi, valid_lo, valid_hi, test_lo)
        if split is None:
            continue
        if mid >= test_lo and mid > split_cfg.get("test_hi", mid):
            continue
        box = pick_primary_box(clip["boxes"].get(mid, []))
        visible = 1 if box is not None else 0
        cx = cy = None
        bbox = None
        if box is not None:
            cx, cy = box_center(box)
            bbox = [round(v, 2) for v in box]
        tid = f"{meta['id']}_f{mid:04d}"
        out.append(
            {
                "id": tid,
                "split": split,
                "clip_id": meta["id"],
                "camera": meta["camera"],
                "match": meta["match"],
                "mid_frame": mid,
                "prev": f"clips/{meta['id']}/frames/{mid - 1:04d}.jpg",
                "mid": f"clips/{meta['id']}/frames/{mid:04d}.jpg",
                "next": f"clips/{meta['id']}/frames/{mid + 1:04d}.jpg",
                "visible": visible,
                "cx": None if cx is None else round(cx, 2),
                "cy": None if cy is None else round(cy, 2),
                "bbox_xywh": bbox,
                "width": clip["width"],
                "height": clip["height"],
                "heatmap": f"heatmaps/{split}/{tid}.npy",
            }
        )
    return out


def write_heatmap(trip: dict, out_root: Path) -> None:
    hm_path = out_root / trip["heatmap"]
    hm_path.parent.mkdir(parents=True, exist_ok=True)
    if trip["visible"] != 1:
        np.save(hm_path, np.zeros((HM_H, HM_W), dtype=np.uint8))
        return
    heat = gaussian_heatmap(trip["cx"], trip["cy"], trip["width"], trip["height"])
    np.save(hm_path, heat)


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")


def summarize(trips: list[dict]) -> dict:
    by = defaultdict(lambda: {"n": 0, "visible": 0})
    for t in trips:
        by[t["split"]]["n"] += 1
        by[t["split"]]["visible"] += int(t["visible"] == 1)
    return dict(by)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=OUT_DEFAULT)
    ap.add_argument("--no-heatmaps", action="store_true")
    ap.add_argument("--clean", action="store_true", help="Delete existing out dir first")
    args = ap.parse_args()
    out = args.out
    if args.clean and out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True, exist_ok=True)

    quad_split = {
        "train_hi": 219,
        "valid_lo": 220,
        "valid_hi": 239,
        "test_lo": 240,
        "test_hi": 10**9,
    }
    m3_split = {
        "train_hi": 99,
        "valid_lo": 100,
        "valid_hi": 119,
        "test_lo": 120,
        "test_hi": 193,
    }

    all_trips = []
    clip_stats = []
    for meta in QUAD_CLIPS:
        clip = load_clip(meta)
        write_clip_labels(out / "clips" / meta["id"], clip)
        trips = build_triplets(clip, quad_split)
        all_trips.extend(trips)
        clip_stats.append(
            {
                "id": meta["id"],
                "camera": meta["camera"],
                "n_frames": len(clip["frames"]),
                "n_visible_frames": len(clip["boxes"]),
                "width": clip["width"],
                "height": clip["height"],
            }
        )
        print(f"clip {meta['id']}: frames={len(clip['frames'])} visible={len(clip['boxes'])} trips={len(trips)}")

    clip = load_clip(MATCH3_CLIP)
    write_clip_labels(out / "clips" / MATCH3_CLIP["id"], clip)
    trips = build_triplets(clip, m3_split)
    # drop match3 mid > test_hi
    trips = [t for t in trips if not (t["split"] == "test" and t["mid_frame"] > m3_split["test_hi"])]
    all_trips.extend(trips)
    clip_stats.append(
        {
            "id": MATCH3_CLIP["id"],
            "camera": MATCH3_CLIP["camera"],
            "n_frames": len(clip["frames"]),
            "n_visible_frames": len(clip["boxes"]),
            "width": clip["width"],
            "height": clip["height"],
        }
    )
    print(f"clip {MATCH3_CLIP['id']}: frames={len(clip['frames'])} visible={len(clip['boxes'])} trips={len(trips)}")

    by_split = defaultdict(list)
    for t in all_trips:
        by_split[t["split"]].append(t)

    for split, rows in by_split.items():
        write_jsonl(out / "splits" / f"{split}_triplets.jsonl", rows)
        if not args.no_heatmaps:
            for t in rows:
                write_heatmap(t, out)
        print(f"{split}: {len(rows)} triplets, visible={sum(r['visible'] for r in rows)}")

    # TrackNet-friendly CSV (mid-frame absolute path style relative to pack)
    for split, rows in by_split.items():
        csv_path = out / "splits" / f"{split}_tracknet.csv"
        with csv_path.open("w", encoding="utf-8") as f:
            f.write("prev,mid,next,visible,x,y\n")
            for t in rows:
                x = "" if t["cx"] is None else f"{t['cx']:.2f}"
                y = "" if t["cy"] is None else f"{t['cy']:.2f}"
                f.write(f"{t['prev']},{t['mid']},{t['next']},{t['visible']},{x},{y}\n")

    manifest = {
        "name": "ball_tracknet_seq_v1",
        "purpose": "Side A/B: Train TrackNet/VballNet from continuous gold only",
        "format": {
            "input": "3 consecutive RGB frames (prev, mid, next)",
            "target": f"uint8 heatmap {HM_W}x{HM_H} (sigma={SIGMA}) for mid frame",
            "coords": "full-frame pixel centers (cx, cy) on review JPGs",
        },
        "sources": clip_stats,
        "splits": summarize(all_trips),
        "holdouts": {
            "4quad": "train mid 1-219, valid 220-239, test 240+",
            "match3_p10_31": "train mid 1-99, valid 100-119, test 120-193",
            "excluded": [
                "match3_quad_p8_87 (not human-reviewed)",
                "sparse Match1/Match2 train packs",
                "Gold100 / match2_gold_frames",
            ],
        },
        "builder": "scripts/gold_set/build_tracknet_seq_pack.py",
        "heatmaps": not args.no_heatmaps,
        "heatmap_size": [HM_W, HM_H],
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"wrote {out}")
    print(json.dumps(manifest["splits"], indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
