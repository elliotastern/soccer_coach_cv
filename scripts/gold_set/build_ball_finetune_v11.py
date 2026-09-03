#!/usr/bin/env python3
"""v11 mix: all validated human ball labels, eval packs held out.

Train: Match1 gold + batch3, Match2 train100, strided 4quad Top Left,
       Match3 P10 M1 early labeled frames.
Never train: Gold100, match2_gold_frames, batch2, unvalidated 4quad drafts,
             Match3 M1 frames 120–194, 4quad frames 240–299.
"""
from __future__ import annotations

import importlib.util
import json
import random
import shutil
import sys
import xml.etree.ElementTree as ET
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
MIX_PY = Path("/Volumes/LaCie/Projects/Soccer project data/scripts/build_ball_finetune_match_mix.py")
V9_PY = ROOT / "scripts/gold_set/build_ball_finetune_match2_v9.py"
OUT = Path("/Volumes/LaCie/Projects/Soccer project data/ball_finetune_v11")
MATCH2_TRAIN = ROOT / "data/processed/gold_sets/match2_train_label100"
MATCH2_GOLD = ROOT / "data/processed/gold_sets/match2_gold_frames"
MATCH3 = ROOT / "data/processed/gold_sets/match3_quad_p10_31"
SEED = 42

# Dense 60fps clips → ~12 fps (stride 5). Last 1s of 4quad stays eval.
QUAD_STRIDE = 5
QUAD_TRAIN_MAX = 239
M3_TRAIN_MAX = 119
M3_STRIDE = 2
MATCH1_AUG = 2
MATCH2_AUG = 3
DENSE_AUG = 3
N_OFFICIAL = 40
N_KJOYY = 60
N_VALID_OFFICIAL = 15
N_VALID_KJOYY = 20

QUAD_PACKS = [
    ("match2_4quad_top_left", "quad_p10"),
    ("match2_4quad_top_left_p7", "quad_p7"),
    ("match2_4quad_top_left_cam4plus", "quad_cam4plus"),
]


def load_py(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def clamp_box(box, w: int, h: int) -> list | None:
    x, y, bw, bh = [float(v) for v in box]
    x1 = max(0.0, x)
    y1 = max(0.0, y)
    x2 = min(float(w), x + bw)
    y2 = min(float(h), y + bh)
    nw, nh = x2 - x1, y2 - y1
    if nw < 4 or nh < 4:
        return None
    return [x1, y1, nw, nh]


def frame_jpg(folder: Path, i: int) -> Path | None:
    for name in (f"{i:03d}.jpg", f"{i:04d}.jpg"):
        p = folder / name
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
            xtl, ytl = float(box.get("xtl")), float(box.get("ytl"))
            xbr, ybr = float(box.get("xbr")), float(box.get("ybr"))
            raw[frame].append([xtl, ytl, xbr - xtl, ybr - ytl])
    return dict(raw)


def items_from_quad(pack_name: str, tag: str, w: int, h: int) -> tuple[list, list]:
    pack = ROOT / "data/processed/gold_sets" / pack_name
    xml = pack / "gold" / "annotations.xml"
    frames = pack / "review" / "frames"
    by_f = xml_ball_boxes(xml)
    train, hold = [], []
    for i, boxes in sorted(by_f.items()):
        src = frame_jpg(frames, i)
        if src is None:
            raise FileNotFoundError(f"{pack_name} missing frame {i}")
        kept = [b for b in (clamp_box(b, w, h) for b in boxes) if b]
        if not kept:
            continue
        item = {
            "src": src,
            "width": w,
            "height": h,
            "boxes": kept,
            "source": tag,
            "file_name": f"{pack_name}_{src.name}",
            "frame": i,
        }
        if i <= QUAD_TRAIN_MAX and i % QUAD_STRIDE == 0:
            train.append(item)
        elif i > QUAD_TRAIN_MAX:
            hold.append(item)
    return train, hold


def items_from_match3(w: int = 1920, h: int = 1080) -> tuple[list, list]:
    lab = json.loads((MATCH3 / "labels.json").read_text())
    frames = MATCH3 / "review" / "frames"
    train, hold = [], []
    for fr in lab["frames"]:
        seed = fr["cams"]["P10"]
        balls = seed.get("gt_balls") or []
        if not balls:
            continue
        i = int(fr["i"])
        src = frames / fr["file"]
        if not src.is_file():
            src = frame_jpg(frames, i)
        if src is None or not src.is_file():
            raise FileNotFoundError(f"match3 frame {i}")
        kept = [b for b in (clamp_box([bb["x"], bb["y"], bb["w"], bb["h"]], w, h) for bb in balls) if b]
        if not kept:
            continue
        item = {
            "src": src,
            "width": w,
            "height": h,
            "boxes": kept,
            "source": "match3_p10_31",
            "file_name": f"match3_{src.name}",
            "frame": i,
        }
        if i <= M3_TRAIN_MAX and i % M3_STRIDE == 0:
            train.append(item)
        elif M3_TRAIN_MAX < i <= 194:
            hold.append(item)
    return train, hold


def stills(mix, rng) -> tuple[list, list]:
    official = (
        mix.official_ball_items("train")
        + mix.official_ball_items("valid")
        + mix.official_ball_items("test")
    )
    rng.shuffle(official)
    kjoyy = (
        mix.kjoyy_ball_items("train")
        + mix.kjoyy_ball_items("valid")
        + mix.kjoyy_ball_items("test")
    )
    rng.shuffle(kjoyy)
    train = official[:N_OFFICIAL] + kjoyy[:N_KJOYY]
    valid = (
        official[N_OFFICIAL : N_OFFICIAL + N_VALID_OFFICIAL]
        + kjoyy[N_KJOYY : N_KJOYY + N_VALID_KJOYY]
    )
    return train, valid


def split_match2(mix, v9) -> tuple[list, list]:
    train_names = v9.names_in_coco(
        ROOT / "data/processed/gold_sets/match2_train_test/train/_annotations.coco.json"
    )
    valid_names = v9.names_in_coco(
        ROOT / "data/processed/gold_sets/match2_train_test/valid/_annotations.coco.json"
    )
    gold_keys = v9.keys_from_pack(MATCH2_GOLD)
    match2_all = mix.match_train_ball_items_from(MATCH2_TRAIN, "match2_train100")
    match2_man = json.loads((MATCH2_TRAIN / "manifest.json").read_text())
    by_name = {r["image"]: r for r in match2_man["frames"]}
    train, valid = [], []
    for item in match2_all:
        row = by_name[item["file_name"]]
        key = (row["camera"], int(row["frame_idx"]))
        if key in gold_keys:
            raise RuntimeError(f"Match2 gold leak: {key}")
        if item["file_name"] in valid_names:
            valid.append(item)
        elif item["file_name"] in train_names:
            train.append(item)
    return train, valid


def main() -> int:
    mix = load_py(MIX_PY, "mix")
    v9 = load_py(V9_PY, "v9")
    if OUT.exists():
        shutil.rmtree(OUT)
    OUT.mkdir(parents=True)
    mix.OUT = OUT
    rng = random.Random(SEED)

    match1_raw = [
        it for it in mix.match_train_ball_items()
        if it["source"] != "match1_train_batch2"
    ]
    match2_train, match2_valid = split_match2(mix, v9)
    quad_train, quad_hold = [], []
    for pack, tag in QUAD_PACKS:
        tr, ho = items_from_quad(pack, tag, 1920, 1080)
        quad_train.extend(tr)
        quad_hold.extend(ho)
    m3_train, m3_hold = items_from_match3()

    match1_items = mix.expand_match_with_augs(match1_raw, MATCH1_AUG, rng)
    match2_items = mix.expand_match_with_augs(match2_train, MATCH2_AUG, rng)
    dense_raw = quad_train + m3_train
    dense_items = mix.expand_match_with_augs(dense_raw, DENSE_AUG, rng)
    still_train, still_valid = stills(mix, rng)

    gold100_test = v9.gold100_test_items(mix)
    match2_test = v9.match2_test_items(mix)

    train_src = {m["src"].resolve() for m in match1_raw + match2_train + dense_raw}
    hold_src = {h["src"].resolve() for h in m3_hold + quad_hold}
    if train_src & hold_src:
        raise RuntimeError("train/holdout image leak")
    leak_names = {m["file_name"] for m in match1_raw + match2_train} & {
        t["file_name"] for t in gold100_test + match2_test
    }
    if leak_names:
        raise RuntimeError(f"held-out name leak: {sorted(leak_names)[:5]}")

    train_items = match2_items + match1_items + dense_items + still_train
    valid_items = still_valid + match2_valid
    rng.shuffle(train_items)
    rng.shuffle(valid_items)
    test_items = gold100_test + match2_test

    train_stats = mix.write_split(OUT / "train", train_items, "train")
    valid_stats = mix.write_split(OUT / "valid", valid_items, "valid")
    test_stats = mix.write_split(OUT / "test", test_items, "test")
    manifest = {
        "seed": SEED,
        "recipe": "v11_all_validated",
        "train": train_stats,
        "valid": valid_stats,
        "test": test_stats,
        "raw_counts": {
            "match1": len(match1_raw),
            "match2_train100": len(match2_train),
            "quad_stride": len(quad_train),
            "match3_stride": len(m3_train),
        },
        "holdout_not_in_pack_images": {
            "match3_m1_120_194": len(m3_hold),
            "quad_240_299": len(quad_hold),
        },
        "held_out": {
            "match2_gold_frames": "data/processed/gold_sets/match2_gold_frames",
            "gold100_strip_0_49": "data/processed/gold_sets/match1_1_100",
            "match3_m1_frames": f"{M3_TRAIN_MAX + 1}–194",
            "quad_last_1s": "frames 240–299 of Top Left P10/P7/Cam4plus",
            "unvalidated": [
                "math_1_training_batch2",
                "match2_4quad_center_start_cam4plus",
                "match2_4quad_label",
            ],
        },
        "aug": {"match1": MATCH1_AUG, "match2": MATCH2_AUG, "dense": DENSE_AUG},
        "stride": {"quad": QUAD_STRIDE, "match3": M3_STRIDE},
        "notes": [
            "Validated human boxes only (gold XML / Match3 human labels.json).",
            "Dense 60fps clips strided so near-duplicate frames do not dominate.",
            "Match3 120–194 kept for honest M1 clear-ball R.",
            "Gold100 0–49 and match2_gold_frames remain test-only.",
        ],
    }
    (OUT / "manifest.json").write_text(json.dumps(manifest, indent=2))
    local = ROOT / "data/processed/gold_sets/ball_finetune_v11_manifest.json"
    local.write_text(json.dumps(manifest, indent=2))
    print(json.dumps(manifest, indent=2))
    print(f"Wrote pack {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
