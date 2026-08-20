#!/usr/bin/env python3
"""Sweep TrackNet heatmap-peak gates vs v11 emit@0.80 on seq pack test."""
from __future__ import annotations

import argparse
import json
import sys
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


def tn_peak(model, device, pack: Path, row: dict) -> tuple[float, float, float]:
    w, h = HM_W, HM_H
    prev = load_rgb(pack / row["prev"], (w, h))
    mid = load_rgb(pack / row["mid"], (w, h))
    nxt = load_rgb(pack / row["next"], (w, h))
    x = torch.from_numpy(np.concatenate([prev, mid, nxt], axis=0)).unsqueeze(0).to(device)
    with torch.no_grad():
        heat = model(x).squeeze().cpu().numpy()
    px, py = peak_xy(heat)
    fw, fh = float(row["width"]), float(row["height"])
    return px * (fw / HM_W), py * (fh / HM_H), float(heat.max())


def v11_best(model, mid_path: Path, thr: float = 0.30) -> tuple[float, float, float] | None:
    raw = model.predict(Image.open(mid_path).convert("RGB"), threshold=thr)
    if not hasattr(raw, "class_id") or len(raw.class_id) == 0:
        return None
    best_i, best_c = None, -1.0
    for i in range(len(raw.class_id)):
        conf = float(raw.confidence[i])
        if conf > best_c:
            best_c, best_i = conf, i
    if best_i is None:
        return None
    x1, y1, x2, y2 = map(float, raw.xyxy[best_i])
    return (x1 + x2) / 2.0, (y1 + y2) / 2.0, best_c


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pack", type=Path, default=PACK_DEFAULT)
    ap.add_argument("--tracknet", type=Path, default=TN_DEFAULT)
    ap.add_argument("--v11", type=Path, default=V11_DEFAULT)
    ap.add_argument("--tol-full", type=float, default=15.0)
    ap.add_argument("--split", default="test")
    ap.add_argument(
        "--out",
        type=Path,
        default=ROOT / "reports/tracknet_seq_v1/gated_compare_vs_v11.json",
    )
    args = ap.parse_args()

    from src.perception.rfdetr_local import load_ball_model

    rows = SeqTripletDataset(args.pack, args.split).rows
    device = torch.device("cpu")
    tn = TrackNetV2().to(device)
    tn.load_state_dict(torch.load(args.tracknet, map_location=device, weights_only=False)["model"])
    tn.eval()
    v11 = load_ball_model(str(args.v11))

    samples = []
    for i, row in enumerate(rows):
        if i and i % 25 == 0:
            print(f"  {i}/{len(rows)}", flush=True)
        vis = int(row["visible"]) == 1
        gx, gy = row.get("cx"), row.get("cy")
        tcx, tcy, tmax = tn_peak(tn, device, args.pack, row)
        dist_tn = None if not vis else ((tcx - gx) ** 2 + (tcy - gy) ** 2) ** 0.5
        vp = v11_best(v11, args.pack / row["mid"], 0.30)
        dist_v11 = conf_v11 = None
        if vis and vp is not None:
            vx, vy, conf_v11 = vp
            dist_v11 = ((vx - gx) ** 2 + (vy - gy) ** 2) ** 0.5
        samples.append(
            {
                "clip_id": row["clip_id"],
                "visible": vis,
                "tn_peak": tmax,
                "tn_ok": (dist_tn is not None and dist_tn <= args.tol_full),
                "v11_conf": conf_v11,
                "v11_ok": (dist_v11 is not None and dist_v11 <= args.tol_full),
            }
        )

    n_vis = sum(1 for s in samples if s["visible"])
    gates = [round(x, 2) for x in np.linspace(0.05, 0.95, 19)]
    sweep = []
    for g in gates:
        emit = [s for s in samples if s["tn_peak"] >= g]
        emit_vis = [s for s in emit if s["visible"]]
        hit = sum(1 for s in emit_vis if s["tn_ok"])
        # precision proxy: among emits on labeled frames, fraction correct
        # (almost all test mids are visible; empty count is tiny)
        p_emit = hit / len(emit_vis) if emit_vis else 0.0
        r_clear = hit / n_vis if n_vis else 0.0
        sweep.append(
            {
                "tn_peak_gate": g,
                "n_emit": len(emit),
                "n_emit_vis": len(emit_vis),
                "hit": hit,
                "P_emit_proxy": p_emit,
                "clear_R": r_clear,
            }
        )

    v11_08_hit = sum(1 for s in samples if s["visible"] and s["v11_conf"] is not None and s["v11_conf"] >= 0.80 and s["v11_ok"])
    v11_08_emit = sum(1 for s in samples if s["visible"] and s["v11_conf"] is not None and s["v11_conf"] >= 0.80)
    v11_03_hit = sum(1 for s in samples if s["visible"] and s["v11_ok"])

    # pick TN gate: highest clear_R with P_emit_proxy >= 0.80
    candidates = [r for r in sweep if r["P_emit_proxy"] >= 0.80]
    pick = max(candidates, key=lambda r: r["clear_R"]) if candidates else max(sweep, key=lambda r: r["P_emit_proxy"])

    out = {
        "tol_full_px": args.tol_full,
        "n_vis": n_vis,
        "n_rows": len(samples),
        "v11": {
            "emit08_hit": v11_08_hit,
            "emit08_n": v11_08_emit,
            "emit08_clear_R": v11_08_hit / n_vis if n_vis else 0.0,
            "emit08_P_proxy": v11_08_hit / v11_08_emit if v11_08_emit else 0.0,
            "thr03_clear_R": v11_03_hit / n_vis if n_vis else 0.0,
        },
        "tracknet_ungated_clear_R": sum(1 for s in samples if s["visible"] and s["tn_ok"]) / n_vis,
        "tracknet_sweep": sweep,
        "tracknet_pick_P80": pick,
        "peak_hist_vis": {
            "p50": float(np.median([s["tn_peak"] for s in samples if s["visible"]])),
            "p10": float(np.percentile([s["tn_peak"] for s in samples if s["visible"]], 10)),
            "p90": float(np.percentile([s["tn_peak"] for s in samples if s["visible"]], 90)),
        },
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(json.dumps({"v11": out["v11"], "tn_ungated": out["tracknet_ungated_clear_R"], "tn_pick": pick}, indent=2))
    print("wrote", args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
