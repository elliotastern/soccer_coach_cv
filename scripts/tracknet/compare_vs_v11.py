#!/usr/bin/env python3
"""Fair A/B: TrackNet seq vs RF-DETR v11 on ball_tracknet_seq_v1 test mids."""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from dataset import HM_H, HM_W, SeqTripletDataset, load_rgb
from model import TrackNetV2
from train import peak_xy

PACK_DEFAULT = Path("/Volumes/LaCie/Projects/Soccer project data/ball_tracknet_seq_v1")
TN_DEFAULT = ROOT / "models/tracknet_seq_v1/best.pth"
V11_DEFAULT = ROOT / "models/v11_snaps/post_train/checkpoint.pth"


def full_tol_from_hm(tol_hm: float, width: float) -> float:
    return tol_hm * (width / float(HM_W))


def tracknet_pred(model, device, pack: Path, row: dict) -> tuple[float, float, float]:
    """Return cx,cy in full-res pixels and peak heat value."""
    w, h = HM_W, HM_H
    prev = load_rgb(pack / row["prev"], (w, h))
    mid = load_rgb(pack / row["mid"], (w, h))
    nxt = load_rgb(pack / row["next"], (w, h))
    x = torch.from_numpy(np.concatenate([prev, mid, nxt], axis=0)).unsqueeze(0).to(device)
    with torch.no_grad():
        heat = model(x).squeeze().cpu().numpy()
    px, py = peak_xy(heat)
    fw = float(row["width"])
    fh = float(row["height"])
    return px * (fw / HM_W), py * (fh / HM_H), float(heat.max())


def v11_pred(model, mid_path: Path, thr: float) -> tuple[float, float, float] | None:
    pil = Image.open(mid_path).convert("RGB")
    raw = model.predict(pil, threshold=thr)
    if not hasattr(raw, "class_id") or len(raw.class_id) == 0:
        return None
    best_i = None
    best_c = -1.0
    for i in range(len(raw.class_id)):
        # ball class is typically 1 for our ball-only ckpt; accept any if single-class
        cid = int(raw.class_id[i])
        if cid not in (0, 1) and len(set(int(c) for c in raw.class_id)) > 1:
            continue
        conf = float(raw.confidence[i])
        if conf > best_c:
            best_c = conf
            best_i = i
    if best_i is None:
        return None
    x1, y1, x2, y2 = map(float, raw.xyxy[best_i])
    return (x1 + x2) / 2.0, (y1 + y2) / 2.0, best_c


def score_split(rows, pack, tn_model, tn_device, v11_model, tol_full: float) -> dict:
    by_clip = defaultdict(lambda: {"n_vis": 0, "tn_hit": 0, "v11_hit08": 0, "v11_hit03": 0})
    totals = {"n_vis": 0, "tn_hit": 0, "v11_hit08": 0, "v11_hit03": 0, "tn_empty_ok": 0, "n_empty": 0}
    for i, row in enumerate(rows):
        if i and i % 25 == 0:
            print(f"  scored {i}/{len(rows)} …", flush=True)
        clip = row["clip_id"]
        vis = int(row["visible"]) == 1
        gx, gy = row.get("cx"), row.get("cy")
        tcx, tcy, tmax = tracknet_pred(tn_model, tn_device, pack, row)
        if not vis:
            totals["n_empty"] += 1
            # empty: tracknet peak should be weak
            if tmax < 0.3:
                totals["tn_empty_ok"] += 1
            continue
        totals["n_vis"] += 1
        by_clip[clip]["n_vis"] += 1
        dist_tn = ((tcx - gx) ** 2 + (tcy - gy) ** 2) ** 0.5
        if dist_tn <= tol_full:
            totals["tn_hit"] += 1
            by_clip[clip]["tn_hit"] += 1
        mid_path = pack / row["mid"]
        pred = v11_pred(v11_model, mid_path, 0.30)
        if pred is not None:
            vx, vy, conf = pred
            dist = ((vx - gx) ** 2 + (vy - gy) ** 2) ** 0.5
            if dist <= tol_full:
                totals["v11_hit03"] += 1
                by_clip[clip]["v11_hit03"] += 1
                if conf >= 0.80:
                    totals["v11_hit08"] += 1
                    by_clip[clip]["v11_hit08"] += 1
    out = {
        "tol_full_px": tol_full,
        "totals": {
            **totals,
            "tn_recall": totals["tn_hit"] / totals["n_vis"] if totals["n_vis"] else 0.0,
            "v11_recall_emit08": totals["v11_hit08"] / totals["n_vis"] if totals["n_vis"] else 0.0,
            "v11_recall_thr03": totals["v11_hit03"] / totals["n_vis"] if totals["n_vis"] else 0.0,
        },
        "by_clip": {},
    }
    for clip, c in by_clip.items():
        n = c["n_vis"] or 1
        out["by_clip"][clip] = {
            **c,
            "tn_recall": c["tn_hit"] / n,
            "v11_recall_emit08": c["v11_hit08"] / n,
            "v11_recall_thr03": c["v11_hit03"] / n,
        }
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pack", type=Path, default=PACK_DEFAULT)
    ap.add_argument("--tracknet", type=Path, default=TN_DEFAULT)
    ap.add_argument("--v11", type=Path, default=V11_DEFAULT)
    ap.add_argument("--tol-hm", type=float, default=4.0, help="heatmap pixel tol (TrackNet default)")
    ap.add_argument("--split", default="test")
    ap.add_argument("--out", type=Path, default=ROOT / "reports/tracknet_seq_v1/compare_vs_v11.json")
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    from src.perception.rfdetr_local import load_ball_model

    ds = SeqTripletDataset(args.pack, args.split)
    rows = list(ds.rows)
    if args.limit:
        rows = rows[: args.limit]
    fw = float(rows[0]["width"])
    tol_full = full_tol_from_hm(args.tol_hm, fw)
    print(f"rows={len(rows)} tol_hm={args.tol_hm} → tol_full={tol_full:.1f}px on {fw:.0f}w")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tn = TrackNetV2().to(device)
    ckpt = torch.load(args.tracknet, map_location=device, weights_only=False)
    tn.load_state_dict(ckpt["model"])
    tn.eval()
    print("loaded tracknet", args.tracknet)
    v11 = load_ball_model(str(args.v11))
    print("loaded v11", args.v11)

    metrics = score_split(rows, args.pack, tn, device, v11, tol_full)
    metrics["split"] = args.split
    metrics["tracknet_ckpt"] = str(args.tracknet)
    metrics["v11_ckpt"] = str(args.v11)
    metrics["n_rows"] = len(rows)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(json.dumps(metrics["totals"], indent=2))
    print("by_clip:")
    for k, v in metrics["by_clip"].items():
        print(f"  {k}: tn={v['tn_recall']:.3f} v11@0.8={v['v11_recall_emit08']:.3f} v11@0.3={v['v11_recall_thr03']:.3f} (n={v['n_vis']})")
    print("wrote", args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
