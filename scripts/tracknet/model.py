#!/usr/bin/env python3
"""Minimal TrackNetV2-style U-Net for 3-frame ball heatmaps (MIT-style architecture)."""
from __future__ import annotations

import torch
import torch.nn as nn


def _conv_block(cin: int, cout: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(cin, cout, 3, padding=1, bias=False),
        nn.BatchNorm2d(cout),
        nn.ReLU(inplace=True),
        nn.Conv2d(cout, cout, 3, padding=1, bias=False),
        nn.BatchNorm2d(cout),
        nn.ReLU(inplace=True),
    )


class TrackNetV2(nn.Module):
    """9-ch RGB triplet → 1-ch heatmap (288×512 typical)."""

    def __init__(self, in_channels: int = 9, out_channels: int = 1):
        super().__init__()
        self.down1 = _conv_block(in_channels, 64)
        self.pool1 = nn.MaxPool2d(2)
        self.down2 = _conv_block(64, 128)
        self.pool2 = nn.MaxPool2d(2)
        self.down3 = _conv_block(128, 256)
        self.pool3 = nn.MaxPool2d(2)
        self.mid = _conv_block(256, 512)
        self.up3 = nn.ConvTranspose2d(512, 256, 2, stride=2)
        self.dec3 = _conv_block(512, 256)
        self.up2 = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.dec2 = _conv_block(256, 128)
        self.up1 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.dec1 = _conv_block(128, 64)
        self.head = nn.Conv2d(64, out_channels, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        d1 = self.down1(x)
        d2 = self.down2(self.pool1(d1))
        d3 = self.down3(self.pool2(d2))
        m = self.mid(self.pool3(d3))
        u3 = self.dec3(torch.cat([self.up3(m), d3], dim=1))
        u2 = self.dec2(torch.cat([self.up2(u3), d2], dim=1))
        u1 = self.dec1(torch.cat([self.up1(u2), d1], dim=1))
        return torch.sigmoid(self.head(u1))
