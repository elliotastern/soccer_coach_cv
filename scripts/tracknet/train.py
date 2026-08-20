#!/usr/bin/env python3
"""Train TrackNetV2-style model on ball_tracknet_seq_v1."""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parent))
from dataset import HM_H, HM_W, SeqTripletDataset
from model import TrackNetV2


def peak_xy(heat: np.ndarray) -> tuple[float, float]:
    flat = heat.reshape(-1)
    idx = int(flat.argmax())
    y, x = divmod(idx, heat.shape[1])
    return float(x), float(y)


def heatmap_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """MSE with heavy weight on ball pixels so the net cannot collapse to zeros."""
    weight = 1.0 + target * 99.0
    return (weight * (pred - target) ** 2).mean()


def eval_loader(model, loader, device, tol: float) -> dict:
    model.eval()
    n = 0
    hit = 0
    vis_n = 0
    loss_sum = 0.0
    crit = nn.MSELoss(reduction="sum")
    with torch.no_grad():
        for batch in loader:
            x = batch["x"].to(device)
            y = batch["y"].to(device)
            pred = model(x)
            loss_sum += float(crit(pred, y).item())
            n += x.size(0)
            pred_np = pred.squeeze(1).cpu().numpy()
            for i in range(x.size(0)):
                if int(batch["visible"][i]) != 1:
                    continue
                vis_n += 1
                px, py = peak_xy(pred_np[i])
                gx = float(batch["cx"][i]) * (HM_W / float(batch["width"][i]))
                gy = float(batch["cy"][i]) * (HM_H / float(batch["height"][i]))
                if (px - gx) ** 2 + (py - gy) ** 2 <= tol**2:
                    hit += 1
    return {
        "loss": loss_sum / max(n, 1),
        "n": n,
        "visible": vis_n,
        "precision_tol": (hit / vis_n) if vis_n else 0.0,
        "hit": hit,
    }


def save_ckpt(path: Path, model, opt, epoch: int, best: float, args) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "epoch": epoch,
            "model": model.state_dict(),
            "opt": opt.state_dict(),
            "best_prec": best,
            "args": vars(args),
        },
        path,
    )


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pack", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--tol", type=float, default=4.0)
    ap.add_argument("--resume", type=Path, default=None)
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_ds = SeqTripletDataset(args.pack, "train")
    valid_ds = SeqTripletDataset(args.pack, "valid")
    test_ds = SeqTripletDataset(args.pack, "test")
    train_loader = DataLoader(
        train_ds, batch_size=args.batch, shuffle=True, num_workers=args.workers, pin_memory=True
    )
    valid_loader = DataLoader(valid_ds, batch_size=args.batch, shuffle=False, num_workers=args.workers)
    test_loader = DataLoader(test_ds, batch_size=args.batch, shuffle=False, num_workers=args.workers)

    model = TrackNetV2().to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)
    start_epoch = 1
    best = -1.0
    if args.resume and args.resume.is_file():
        ckpt = torch.load(args.resume, map_location=device)
        model.load_state_dict(ckpt["model"])
        opt.load_state_dict(ckpt["opt"])
        start_epoch = int(ckpt.get("epoch", 0)) + 1
        best = float(ckpt.get("best_prec", -1.0))
        print(f"resumed epoch {start_epoch - 1} best_prec={best:.4f}")

    args.out.mkdir(parents=True, exist_ok=True)
    log_path = args.out / "train_log.jsonl"
    print(
        f"device={device} train={len(train_ds)} valid={len(valid_ds)} test={len(test_ds)} "
        f"batch={args.batch} epochs={args.epochs}"
    )

    for epoch in range(start_epoch, args.epochs + 1):
        model.train()
        t0 = time.time()
        loss_sum = 0.0
        n = 0
        for batch in train_loader:
            x = batch["x"].to(device, non_blocking=True)
            y = batch["y"].to(device, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            pred = model(x)
            loss = heatmap_loss(pred, y)
            loss.backward()
            opt.step()
            loss_sum += float(loss.item()) * x.size(0)
            n += x.size(0)
        train_loss = loss_sum / max(n, 1)
        val = eval_loader(model, valid_loader, device, args.tol)
        row = {
            "epoch": epoch,
            "train_loss": train_loss,
            "valid_loss": val["loss"],
            "valid_prec_tol": val["precision_tol"],
            "valid_hit": val["hit"],
            "valid_visible": val["visible"],
            "sec": round(time.time() - t0, 1),
        }
        with log_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(row) + "\n")
        print(
            f"epoch {epoch}/{args.epochs} train_loss={train_loss:.5f} "
            f"valid_prec@{args.tol}px={val['precision_tol']:.3f} "
            f"({val['hit']}/{val['visible']}) {row['sec']}s"
        )
        save_ckpt(args.out / "last.pth", model, opt, epoch, best, args)
        if val["precision_tol"] >= best:
            best = val["precision_tol"]
            save_ckpt(args.out / "best.pth", model, opt, epoch, best, args)
            print(f"  saved best.pth prec={best:.3f}")

    # final test on best
    ckpt = torch.load(args.out / "best.pth", map_location=device)
    model.load_state_dict(ckpt["model"])
    test = eval_loader(model, test_loader, device, args.tol)
    summary = {
        "best_valid_prec_tol": best,
        "test_prec_tol": test["precision_tol"],
        "test_hit": test["hit"],
        "test_visible": test["visible"],
        "tol_px": args.tol,
        "heatmap": [HM_W, HM_H],
    }
    (args.out / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print("DONE", json.dumps(summary))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
