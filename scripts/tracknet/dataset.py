#!/usr/bin/env python3
"""Dataset for ball_tracknet_seq_v1 pack (CSV or jsonl triplets)."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

HM_W = 512
HM_H = 288


def load_rgb(path: Path, size: tuple[int, int]) -> np.ndarray:
    with Image.open(path) as im:
        im = im.convert("RGB").resize(size, Image.BILINEAR)
        arr = np.asarray(im, dtype=np.float32) / 255.0
    return arr.transpose(2, 0, 1)


def load_heatmap(path: Path | None, visible: int, cx, cy, fw: int, fh: int) -> np.ndarray:
    if path is not None and path.is_file():
        heat = np.load(path).astype(np.float32) / 255.0
        if heat.shape != (HM_H, HM_W):
            raise ValueError(f"bad heatmap shape {heat.shape} at {path}")
        return heat
    heat = np.zeros((HM_H, HM_W), dtype=np.float32)
    if visible != 1 or cx is None or cy is None:
        return heat
    sx = float(cx) * (HM_W / float(fw))
    sy = float(cy) * (HM_H / float(fh))
    xs = np.arange(HM_W, dtype=np.float32)
    ys = np.arange(HM_H, dtype=np.float32)
    xx, yy = np.meshgrid(xs, ys)
    heat = np.exp(-((xx - sx) ** 2 + (yy - sy) ** 2) / (2.0 * 2.5**2))
    return heat


def read_rows(pack: Path, split: str) -> list[dict]:
    jsonl = pack / "splits" / f"{split}_triplets.jsonl"
    if jsonl.is_file():
        return [json.loads(line) for line in jsonl.read_text().splitlines() if line.strip()]
    csv_path = pack / "splits" / f"{split}_tracknet.csv"
    rows = []
    for i, line in enumerate(csv_path.read_text().splitlines()):
        if i == 0:
            continue
        prev, mid, nxt, vis, x, y = line.split(",")
        rows.append(
            {
                "prev": prev,
                "mid": mid,
                "next": nxt,
                "visible": int(vis),
                "cx": None if x == "" else float(x),
                "cy": None if y == "" else float(y),
                "width": 1920,
                "height": 1080,
                "heatmap": None,
            }
        )
    return rows


class SeqTripletDataset(Dataset):
    def __init__(self, pack: Path, split: str, size: tuple[int, int] = (HM_W, HM_H)):
        self.pack = Path(pack)
        self.size = size  # (W, H)
        self.rows = read_rows(self.pack, split)
        if not self.rows:
            raise FileNotFoundError(f"no rows for {split} in {pack}")

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int) -> dict:
        row = self.rows[idx]
        w, h = self.size
        prev = load_rgb(self.pack / row["prev"], (w, h))
        mid = load_rgb(self.pack / row["mid"], (w, h))
        nxt = load_rgb(self.pack / row["next"], (w, h))
        x = np.concatenate([prev, mid, nxt], axis=0)
        hm_rel = row.get("heatmap")
        hm_path = self.pack / hm_rel if hm_rel else None
        heat = load_heatmap(
            hm_path,
            int(row["visible"]),
            row.get("cx"),
            row.get("cy"),
            int(row.get("width") or 1920),
            int(row.get("height") or 1080),
        )
        return {
            "x": torch.from_numpy(x),
            "y": torch.from_numpy(heat).unsqueeze(0),
            "visible": torch.tensor(int(row["visible"]), dtype=torch.int64),
            "cx": torch.tensor(-1.0 if row.get("cx") is None else float(row["cx"])),
            "cy": torch.tensor(-1.0 if row.get("cy") is None else float(row["cy"])),
            "width": torch.tensor(float(row.get("width") or 1920)),
            "height": torch.tensor(float(row.get("height") or 1080)),
        }
