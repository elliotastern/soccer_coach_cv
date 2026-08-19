"""Score Match 3 xy+fuse plan and implementation (need 9+/10 each)."""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
PLAN = ROOT / "docs/product/MATCH3_XY_BALL_PLAN.md"
OUT = ROOT / "reports/eval_match3/xy_fuse_eng_loop"
PASS = 9.0


def clamp(score: float) -> float:
    return round(max(0.0, min(10.0, score)), 1)


def score_plan() -> tuple[float, list[str]]:
    notes = []
    score = 10.0
    text = PLAN.read_text(encoding="utf-8") if PLAN.is_file() else ""
    if not text:
        return 0.0, ["plan missing"]
    need = [
        ("manual 4-click", "calib"),
        ("Pitch 1", "pitch1"),
        ("bbox foot", "foot"),
        ("convex hull", "local H"),
        ("4 m", "agree m"),
        ("0.80", "emit"),
        ("do not average", "no midpoint"),
        ("FOV", "no FOV when H"),
        ("video title", "cam id"),
        ("Phase 2", "no temporal fusion"),
    ]
    low = text.lower()
    for needle, label in need:
        if needle.lower() not in low:
            score -= 1.0
            notes.append(f"plan missing {label}")
    return clamp(score), notes


def score_roundtrip() -> tuple[float, list[str]]:
    from src.mapping.match3_xy import MATCH3_CAMS, apply_H, load_calib

    notes = []
    score = 10.0
    for cam in MATCH3_CAMS:
        rec = load_calib(cam)
        if rec is None:
            score -= 1.5
            notes.append(f"no H {cam}")
            continue
        worst = 0.0
        for img, pitch in zip(rec["image_points"], rec["pitch_points"]):
            got = apply_H(rec["H"], img[0], img[1])
            err = math.hypot(got[0] - pitch[0], got[1] - pitch[1])
            worst = max(worst, err)
        if worst > 0.15:
            score -= 2
            notes.append(f"{cam} roundtrip {worst:.2f}m")
    return clamp(score), notes


def score_map_rules() -> tuple[float, list[str]]:
    from src.mapping.match3_xy import bbox_foot, hull_support, load_calib, map_ball_box
    from src.mapping.pitch_bounds import in_pitch_bounds

    notes = []
    score = 10.0
    fx, fy = bbox_foot([100, 80, 20, 40])
    if abs(fx - 110) > 1e-6 or abs(fy - 120) > 1e-6:
        score -= 3
        notes.append("bbox is not foot")
    if in_pitch_bounds(10.0, 40.0, margin_m=1.0):
        score -= 3
        notes.append("off-pitch not gated")
    rec = load_calib("P9")
    if rec:
        sup = hull_support(1900, 10, rec["image_points"])
        row = map_ball_box(rec, [1860, 20, 20, 20], 0.99, rec["image_wh"])
        if sup >= 0.35 and row is not None:
            score -= 3
            notes.append("far-from-hull still mapped")
    return clamp(score), notes


def score_fuse() -> tuple[float, list[str]]:
    from src.mapping.match3_xy import fuse_balls

    notes = []
    score = 10.0
    agree = fuse_balls(
        [
            {"cam": "P9", "xy": (36.0, -20.0), "conf": 0.62, "support": 0.9, "weight": 0.56},
            {"cam": "P8", "xy": (38.0, -21.0), "conf": 0.55, "support": 0.2, "weight": 0.11},
        ]
    )
    if not agree or not agree.get("agree"):
        score -= 4
        notes.append("2-cam cluster not fused")
    elif abs(agree["xy"][0] - 37.0) > 0.25:
        score -= 2
        notes.append("fuse used pixel-size not median")
    split = fuse_balls(
        [
            {"cam": "P9", "xy": (40.0, -20.0), "conf": 0.91, "support": 1.0, "weight": 0.91},
            {"cam": "P10", "xy": (-40.0, 20.0), "conf": 0.90, "support": 0.4, "weight": 0.36},
        ]
    )
    if split is None or split.get("agree") or abs(split["xy"][0] - 40.0) > 0.01:
        score -= 4
        notes.append("disagree averaged")
    if fuse_balls(
        [{"cam": "P9", "xy": (36.0, -20.0), "conf": 0.40, "support": 1.0, "weight": 0.40}]
    ) is not None:
        score -= 3
        notes.append("0.4 singleton emitted")
    return clamp(score), notes


def score_ids() -> tuple[float, list[str]]:
    from src.mapping.match3_xy import MATCH3_CAMS, load_calib

    notes = []
    score = 10.0
    if "P12" in MATCH3_CAMS:
        score -= 3
        notes.append("P12 is Match 2, not Match 3")
    if "P9" not in MATCH3_CAMS or "P_Goal1" not in MATCH3_CAMS:
        score -= 3
        notes.append("Match 3 pool incomplete")
    if load_calib("P1") is None:
        score -= 2
        notes.append("P1 H missing")
    return clamp(score), notes


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    keys = ["plan", "roundtrip", "map", "fuse", "cam_ids"]
    scores, notes = {}, {}
    scores["plan"], notes["plan"] = score_plan()
    scores["roundtrip"], notes["roundtrip"] = score_roundtrip()
    scores["map"], notes["map"] = score_map_rules()
    scores["fuse"], notes["fuse"] = score_fuse()
    scores["cam_ids"], notes["cam_ids"] = score_ids()
    fails = [f"{k}={scores[k]} {notes[k]}" for k in keys if scores[k] < PASS]
    summary = {"scores": scores, "notes": notes, "pass": PASS, "fails": fails}
    (OUT / "scores.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(scores)
    if fails:
        print("BELOW 9")
        for f in fails:
            print(" ", f)
        return 1
    print("all subgoals >= 9/10")
    print(f"wrote {OUT / 'scores.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
