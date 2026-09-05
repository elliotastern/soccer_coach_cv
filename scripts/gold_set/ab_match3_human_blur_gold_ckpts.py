#!/usr/bin/env python3
"""Report-only: score ball ckpts v12–v22 on match3_human_blur_gold (box IoU / center).

Not a promote gate. Product checkpoint remains v16 until a dedicated blur residual wins
on this bank without killing clear-strip A/B.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import cv2

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts" / "gold_set"))

from src.perception.ball_prelabel import BallPrelabelConfig, BallPrelabeler  # noqa: E402
from src.perception.rfdetr_local import load_ball_model  # noqa: E402

GOLD = ROOT / "data/processed/gold_sets/match3_human_blur_gold"
FRAMES = GOLD / "frames"
HUMAN = GOLD / "gold" / "human_labels.json"
OUT = ROOT / "reports/eval_match3/improve_eng_loop/ab_match3_human_blur_gold_ckpts.json"
THR = 0.10
IOU_OK = 0.30
IOU_TIGHT = 0.50
CTR_OK = 20.0
CTR_LOOSE = 40.0

# Prefer post_train; skip missing (e.g. v13 on Catch).
CKPT_CANDIDATES = {
    "v12": ["models/v12_hard_snaps/post_train/checkpoint.pth", "models/v12_hard_snaps/checkpoint.pth"],
    "v13": ["models/v13_residual_snaps/post_train/checkpoint.pth", "models/v13_residual_snaps/checkpoint.pth"],
    "v14": ["models/v14_residual_snaps/post_train/checkpoint.pth", "models/v14_residual_snaps/checkpoint.pth"],
    "v15": ["models/v15_residual_snaps/post_train/checkpoint.pth", "models/v15_residual_snaps/checkpoint.pth"],
    "v16": ["models/v16_residual_snaps/post_train/checkpoint.pth", "models/v16_residual_snaps/checkpoint.pth"],
    "v17": ["models/v17_residual_snaps/post_train/checkpoint.pth", "models/v17_residual_snaps/checkpoint.pth"],
    "v18": ["models/v18_residual_snaps/post_train/checkpoint.pth", "models/v18_residual_snaps/checkpoint.pth"],
    "v19": ["models/v19_residual_snaps/post_train/checkpoint.pth", "models/v19_residual_snaps/checkpoint.pth"],
    "v20": ["models/v20_residual_snaps/post_train/checkpoint.pth", "models/v20_residual_snaps/checkpoint.pth"],
    "v21": ["models/v21_residual_snaps/post_train/checkpoint.pth", "models/v21_residual_snaps/checkpoint.pth"],
    "v22": ["models/v22_blur_residual_snaps/post_train/checkpoint.pth", "models/v22_blur_residual_snaps/checkpoint.pth"],
}


def resolve_ckpt(name: str) -> Path | None:
    for rel in CKPT_CANDIDATES[name]:
        path = ROOT / rel
        if path.is_file():
            return path
    return None


def iou_xywh(a: list[float], b: list[float]) -> float:
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    ax2, ay2 = ax + aw, ay + ah
    bx2, by2 = bx + bw, by + bh
    ix1, iy1 = max(ax, bx), max(ay, by)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    union = aw * ah + bw * bh - inter
    return float(inter / union) if union > 0 else 0.0


def center_dist(a: list[float], b: list[float]) -> float:
    return float(
        ((a[0] + a[2] / 2) - (b[0] + b[2] / 2)) ** 2
        + ((a[1] + a[3] / 2) - (b[1] + b[3] / 2)) ** 2
    ) ** 0.5


def gt_box(item: dict) -> list[float]:
    b = item["gt_balls"][0]
    return [float(b["x"]), float(b["y"]), float(b["w"]), float(b["h"])]


def best_det(pre: BallPrelabeler, frame, gt: list[float]) -> dict:
    dets = pre.detect_bgr(frame) or []
    rows = []
    for d in dets:
        box = [float(v) for v in d.bbox]
        conf = float(d.confidence)
        if conf < THR:
            continue
        rows.append(
            {
                "box": box,
                "conf": conf,
                "iou": iou_xywh(gt, box),
                "ctr": center_dist(gt, box),
            }
        )
    if not rows:
        return {"hit": False, "iou": 0.0, "ctr": None, "conf": None, "n_det": 0}
    rows.sort(key=lambda r: (-r["iou"], -r["conf"]))
    top = rows[0]
    return {
        "hit": True,
        "iou": round(top["iou"], 4),
        "ctr": round(top["ctr"], 2),
        "conf": round(top["conf"], 4),
        "n_det": len(rows),
    }


def summarize(rows: list[dict]) -> dict:
    n = len(rows)
    if n == 0:
        return {"n": 0}
    def rate(pred):
        return round(sum(1 for r in rows if pred(r)) / n, 4)

    return {
        "n": n,
        "R_iou30": rate(lambda r: (r.get("iou") or 0) >= IOU_OK),
        "R_iou50": rate(lambda r: (r.get("iou") or 0) >= IOU_TIGHT),
        "R_ctr20": rate(lambda r: r.get("ctr") is not None and r["ctr"] <= CTR_OK),
        "R_ctr40": rate(lambda r: r.get("ctr") is not None and r["ctr"] <= CTR_LOOSE),
        "miss": rate(lambda r: not r.get("hit")),
    }


def subset(rows: list[dict], key: str) -> list[dict]:
    if key == "all":
        return rows
    if key == "blurry":
        return [r for r in rows if r.get("blurry")]
    if key == "streaky":
        return [r for r in rows if r.get("streaky")]
    if key == "clear":
        return [r for r in rows if r.get("clear")]
    if key == "soft":
        return [r for r in rows if not r.get("clear")]
    return rows


def eval_ckpt(name: str, ckpt: Path, items: list[dict]) -> dict:
    print(f"load {name} {ckpt}", flush=True)
    model = load_ball_model(str(ckpt))
    pre = BallPrelabeler(
        model,
        BallPrelabelConfig(
            threshold=THR,
            use_sahi=False,
            topk=5,
            use_size_filter=True,
            min_side=4,
            max_side=240,
            use_kalman=False,
        ),
        class_id=1,
    )
    rows = []
    for i, item in enumerate(items):
        path = FRAMES / item["file"]
        if not path.is_file():
            raise FileNotFoundError(path)
        frame = cv2.imread(str(path))
        if frame is None:
            raise RuntimeError(f"bad image {path}")
        gt = gt_box(item)
        hit = best_det(pre, frame, gt)
        rows.append(
            {
                **hit,
                "pack": item["pack"],
                "frame_i": item["frame_i"],
                "blurry": bool(item.get("blurry")),
                "streaky": bool(item.get("streaky")),
                "clear": bool(item.get("clear")),
            }
        )
        if (i + 1) % 50 == 0:
            print(f"  {name} {i+1}/{len(items)}", flush=True)
    by = {k: summarize(subset(rows, k)) for k in ("all", "blurry", "streaky", "clear", "soft")}
    return {"ckpt": str(ckpt.relative_to(ROOT)), "thr": THR, "by_subset": by}


def main() -> int:
    import argparse

    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--ckpts", nargs="+", default=list(CKPT_CANDIDATES.keys()))
    p.add_argument("--out", type=Path, default=OUT)
    args = p.parse_args()
    payload = json.loads(HUMAN.read_text(encoding="utf-8"))
    items = payload["items"]
    if not items:
        raise SystemExit("empty human_labels.json")
    if not FRAMES.is_dir():
        raise SystemExit(f"missing frames dir {FRAMES}")

    results = {}
    missing = []
    for name in args.ckpts:
        ckpt = resolve_ckpt(name)
        if ckpt is None:
            missing.append(name)
            print(f"skip missing {name}", flush=True)
            continue
        results[name] = eval_ckpt(name, ckpt, items)

    table = []
    for name, row in results.items():
        b = row["by_subset"]
        table.append(
            {
                "ckpt": name,
                "all_R30": b["all"].get("R_iou30"),
                "blur_R30": b["blurry"].get("R_iou30"),
                "streak_R30": b["streaky"].get("R_iou30"),
                "clear_R30": b["clear"].get("R_iou30"),
                "all_ctr20": b["all"].get("R_ctr20"),
                "blur_ctr20": b["blurry"].get("R_ctr20"),
            }
        )

    out = {
        "role": "report_only_human_blur_gold",
        "promote": False,
        "n_items": len(items),
        "by_pack": payload.get("by_pack"),
        "n_blurry": payload.get("n_blurry"),
        "n_streaky": payload.get("n_streaky"),
        "missing_ckpts": missing,
        "metrics": {
            "thr": THR,
            "iou_ok": IOU_OK,
            "iou_tight": IOU_TIGHT,
            "ctr_ok_px": CTR_OK,
            "ctr_loose_px": CTR_LOOSE,
        },
        "table": table,
        "results": results,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(json.dumps(table, indent=2))
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
