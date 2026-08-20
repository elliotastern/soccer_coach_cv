#!/usr/bin/env python3
"""Eval a TrackNet ckpt on pack test/valid split; write metrics JSON."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parent))
from dataset import SeqTripletDataset
from model import TrackNetV2
from train import eval_loader


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pack", type=Path, required=True)
    ap.add_argument("--ckpt", type=Path, required=True)
    ap.add_argument("--split", default="test")
    ap.add_argument("--tol", type=float, default=4.0)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ds = SeqTripletDataset(args.pack, args.split)
    loader = DataLoader(ds, batch_size=args.batch, shuffle=False, num_workers=2)
    model = TrackNetV2().to(device)
    ckpt = torch.load(args.ckpt, map_location=device)
    model.load_state_dict(ckpt["model"])
    metrics = eval_loader(model, loader, device, args.tol)
    metrics["split"] = args.split
    metrics["tol"] = args.tol
    metrics["ckpt"] = str(args.ckpt)
    print(json.dumps(metrics, indent=2))
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
