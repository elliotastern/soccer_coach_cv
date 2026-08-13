#!/usr/bin/env python3
"""Assemble Match 2 gold frames from accepted50 + latest gold XML labels."""
from __future__ import annotations

import json
import re
import shutil
import sys
import xml.etree.ElementTree as ET
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "data/processed/gold_sets/match2_large_ball_accepted50"
OUT = ROOT / "data/processed/gold_sets/match2_gold_frames"
SRC_EDITOR = ROOT / "data/processed/gold_sets/match1_1_100/review/editor.html"
DISPLAY_NAME = "Match 2 gold frames"


def count_gold(xml_path: Path) -> dict:
    root = ET.parse(xml_path).getroot()
    boxes = [b for b in root.findall(".//box") if b.get("outside") != "1"]
    tracks = root.findall(".//track")
    by_frame: dict[int, int] = {}
    for b in boxes:
        f = int(b.get("frame", "0"))
        by_frame[f] = by_frame.get(f, 0) + 1
    return {
        "n_tracks": len(tracks),
        "n_boxes": len(boxes),
        "n_frames_with_box": len(by_frame),
        "n_multiball_frames": sum(1 for n in by_frame.values() if n > 1),
        "labels": dict(Counter(t.get("label") for t in tracks)),
        "max_boxes_per_frame": max(by_frame.values()) if by_frame else 0,
    }


def copy_tree_files(src: Path, dst: Path):
    dst.mkdir(parents=True, exist_ok=True)
    for p in src.iterdir():
        if p.name.startswith("._"):
            continue
        if p.is_file():
            shutil.copy2(p, dst / p.name)


def build_editor(n_frames: int):
    html = SRC_EDITOR.read_text(encoding="utf-8")
    html = html.replace("match1_1_100", "match2_gold_frames")
    names = [f"{i:03d}.jpg" for i in range(n_frames)]
    m = re.search(r"const GOLD100 = \{.*?maxFrame:\s*\d+\s*,\s*\};", html, re.S)
    if not m:
        raise RuntimeError("GOLD100 block missing in template")
    files_js = ",\n                ".join(json.dumps(n) for n in names)
    last = n_frames - 1
    new_block = f"""const GOLD100 = {{
            files: [
                {files_js}
            ],
            base: '/data/processed/gold_sets/match2_gold_frames/review/frames/',
            width: 1920,
            height: 1080,
            maxFrame: {last},
        }};"""
    html = html[: m.start()] + new_block + html[m.end() :]
    html = html.replace('max="99"', f'max="{last}"')
    html = html.replace("seekToFrame(99)", f"seekToFrame({last})")
    html = html.replace("if (currentFrame < 99)", f"if (currentFrame < {last})")
    html = html.replace(" / 99", f" / {last}")
    cm = re.search(r"\.container\s*\{[^}]*height:\s*([^;]+);", html)
    if not cm or "100vh" not in cm.group(1):
        raise RuntimeError(f"CSS corrupted: container height={cm.group(1) if cm else None}")
    ib = re.search(r"\.icon-button\s*\{[^}]*width:\s*([^;]+);", html)
    if ib and "1920" in ib.group(1):
        raise RuntimeError("CSS corrupted: icon-button width")
    out = OUT / "review" / "editor.html"
    out.write_text(html, encoding="utf-8")
    return out


def main() -> int:
    gold_xml = SRC / "gold" / "annotations.xml"
    if not gold_xml.is_file():
        raise SystemExit(f"missing labeled gold XML: {gold_xml}")
    src_man = json.loads((SRC / "manifest.json").read_text())
    n = int(src_man["n_frames"])
    stats = count_gold(gold_xml)
    if stats["n_frames_with_box"] != n:
        print(
            f"WARN: gold has boxes on {stats['n_frames_with_box']} frames, pack has {n}",
            file=sys.stderr,
        )

    if OUT.exists():
        shutil.rmtree(OUT)
    for sub in ("images", "review/frames", "gold", "prelabels"):
        (OUT / sub).mkdir(parents=True, exist_ok=True)

    copy_tree_files(SRC / "images", OUT / "images")
    copy_tree_files(SRC / "review" / "frames", OUT / "review" / "frames")
    shutil.copy2(gold_xml, OUT / "gold" / "annotations.xml")
    # Keep original model proposals as prelabels for comparison
    if (SRC / "prelabels" / "annotations.xml").is_file():
        shutil.copy2(SRC / "prelabels" / "annotations.xml", OUT / "prelabels" / "annotations.xml")
    if (SRC / "prelabels" / "annotations.coco.json").is_file():
        shutil.copy2(
            SRC / "prelabels" / "annotations.coco.json",
            OUT / "prelabels" / "annotations.coco.json",
        )
    if (SRC / "review" / "preds.json").is_file():
        shutil.copy2(SRC / "review" / "preds.json", OUT / "review" / "preds.json")
    if (SRC / "keep_source.json").is_file():
        shutil.copy2(SRC / "keep_source.json", OUT / "keep_source.json")

    editor = build_editor(n)

    frames = []
    for row in src_man["frames"]:
        frames.append({
            **row,
            "stratum": "match2_gold",
            "collection": DISPLAY_NAME,
        })

    manifest = {
        "name": "match2_gold_frames",
        "display_name": DISPLAY_NAME,
        "n_frames": n,
        "sources": {
            "accepted_pack": str(SRC.relative_to(ROOT)),
            "harvest_keep": "data/processed/gold_sets/match2_large_ball_harvest/keep.json",
            "labeled_gold_xml": "data/processed/gold_sets/match2_large_ball_accepted50/gold/annotations.xml",
        },
        "accepted_source_indices": src_man.get("accepted_source_indices", []),
        "label_stats": stats,
        "frames": frames,
        "editor": "/data/processed/gold_sets/match2_gold_frames/review/editor.html",
        "short_url": "http://127.0.0.1:8080/match2-gold",
    }
    (OUT / "manifest.json").write_text(json.dumps(manifest, indent=2))
    (OUT / "README.md").write_text(
        f"""# {DISPLAY_NAME}

{n} Match 2 frames: harvest accepts + manual ball labels (including second balls).

| | |
|--|--|
| Frames | {n} |
| Ball boxes | {stats['n_boxes']} |
| Multi-ball frames | {stats['n_multiball_frames']} |
| Source accepts | `match2_large_ball_harvest` keep.json |
| Labels | `match2_large_ball_accepted50/gold/annotations.xml` |

## Open

```bash
python3 serve_viewer.py --port 8080
# http://127.0.0.1:8080/match2-gold
```

Ball → **N** → draw · **Save** writes `gold/annotations.xml`.
"""
    )
    print(f"Built {OUT}")
    print(f"  frames={n} boxes={stats['n_boxes']} multiball={stats['n_multiball_frames']}")
    print(f"  editor={editor}")
    print("  Open: http://127.0.0.1:8080/match2-gold")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
