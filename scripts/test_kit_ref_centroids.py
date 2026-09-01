#!/usr/bin/env python3
"""Unit tests for labeled kit centroid fit + save/load."""
from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.perception.team_core import (  # noqa: E402
    centroids_from_labeled,
    load_centroids,
    load_kit_ref_meta,
    save_kit_ref,
)


def _feat(blue: float, white: float, yellow: float = 0.0) -> np.ndarray:
    base = np.array([blue, white, yellow, 80.0, 160.0], dtype=np.float32)
    hist = np.zeros(10, dtype=np.float32)
    hist[int(blue * 9)] = 1.0
    return np.concatenate([base, hist])


def test_labeled_centroids():
    t0 = [_feat(0.7, 0.1), _feat(0.65, 0.12)]
    t1 = [_feat(0.1, 0.75), _feat(0.08, 0.8)]
    fit = centroids_from_labeled({0: t0, 1: t1})
    assert fit is not None
    cents, radius = fit
    assert cents.shape == (2, 15)
    assert radius > 0.08
    assert float(cents[0, 0]) > float(cents[1, 0])


def test_save_load_roundtrip():
    t0 = [_feat(0.6, 0.15)]
    t1 = [_feat(0.12, 0.7)]
    fit = centroids_from_labeled({0: t0, 1: t1})
    assert fit is not None
    cents, radius = fit
    with tempfile.TemporaryDirectory() as td:
        path = Path(td) / "team_centroids.json"
        save_kit_ref(path, cents, radius, team_names=("Home", "Away"), n_samples=(1, 1))
        loaded = load_centroids(path)
        assert loaded is not None
        meta = load_kit_ref_meta(path)
        assert meta["team_names"] == ["Home", "Away"]
        np.testing.assert_allclose(loaded[0], cents, rtol=1e-5)
        assert abs(loaded[1] - radius) < 1e-5


if __name__ == "__main__":
    test_labeled_centroids()
    test_save_load_roundtrip()
    print("ok")
