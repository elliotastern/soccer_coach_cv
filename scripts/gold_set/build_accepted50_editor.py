#!/usr/bin/env python3
"""Build draw editor for match2_large_ball_accepted50 (N to add box)."""
from __future__ import annotations

import json
import re
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SRC_EDITOR = ROOT / "data/processed/gold_sets/match1_1_100/review/editor.html"
OUT = ROOT / "data/processed/gold_sets/match2_large_ball_accepted50"


def main() -> int:
    review = OUT / "review"
    names = [f"{i:03d}.jpg" for i in range(50)]
    missing = [n for n in names if not (review / "frames" / n).is_file()]
    if missing:
        raise SystemExit(f"missing review frames: {missing[:5]}")
    if not SRC_EDITOR.is_file():
        raise SystemExit(f"missing template editor: {SRC_EDITOR}")

    html = SRC_EDITOR.read_text(encoding="utf-8")
    html = html.replace("match1_1_100", "match2_large_ball_accepted50")
    m = re.search(
        r"const GOLD100 = \{.*?maxFrame:\s*\d+\s*,\s*\};",
        html,
        re.S,
    )
    if not m:
        raise SystemExit("GOLD100 block not found in template")
    files_js = ",\n                ".join(json.dumps(n) for n in names)
    new_block = f"""const GOLD100 = {{
            files: [
                {files_js}
            ],
            base: '/data/processed/gold_sets/match2_large_ball_accepted50/review/frames/',
            width: 1920,
            height: 1080,
            maxFrame: {len(names) - 1},
        }};"""
    html = html[: m.start()] + new_block + html[m.end() :]
    html = html.replace('max="99"', 'max="49"')
    html = html.replace("seekToFrame(99)", "seekToFrame(49)")
    html = html.replace("if (currentFrame < 99)", "if (currentFrame < 49)")
    html = html.replace(" / 99", " / 49")

    cm = re.search(r"\.container\s*\{[^}]*height:\s*([^;]+);", html)
    if not cm or "100vh" not in cm.group(1):
        raise SystemExit(f"CSS corrupted: container height={cm.group(1) if cm else None}")
    ib = re.search(r"\.icon-button\s*\{[^}]*width:\s*([^;]+);", html)
    if ib and "1920" in ib.group(1):
        raise SystemExit("CSS corrupted: icon-button width")

    (OUT / "gold").mkdir(exist_ok=True)
    gold_xml = OUT / "gold" / "annotations.xml"
    if not gold_xml.is_file():
        shutil.copy2(OUT / "prelabels" / "annotations.xml", gold_xml)

    out = review / "editor.html"
    out.write_text(html, encoding="utf-8")
    print(f"wrote {out}")
    print(
        "Open: http://127.0.0.1:8080/data/processed/gold_sets/"
        "match2_large_ball_accepted50/review/editor.html"
    )
    print("Add box: Ball → N → click-drag")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
