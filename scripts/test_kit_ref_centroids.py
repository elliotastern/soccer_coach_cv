#!/usr/bin/env python3
"""Unit tests for labeled kit centroid fit + save/load + resolve order."""
from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.perception.team_core import (  # noqa: E402
    centroids_from_labeled,
    is_kit_ref,
    load_centroids,
    load_kit_ref_meta,
    resolve_kit_centroids_path,
    save_centroids,
    save_kit_ref,
)
from src.perception.team_tracklet import TrackletTeamModel  # noqa: E402


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
        assert meta["source"] == "kit_label_dashboard"
        assert is_kit_ref(meta)
        np.testing.assert_allclose(loaded[0], cents, rtol=1e-5)
        assert abs(loaded[1] - radius) < 1e-5


def test_resolve_prefers_kit_path():
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        match_level = root / "team_centroids.json"
        run_dir = root / "P10-match4"
        run_dir.mkdir()
        run_path = run_dir / "team_centroids.json"
        cfg_path = root / "cfg_kit.json"
        t0, t1 = [_feat(0.6, 0.1)], [_feat(0.1, 0.7)]
        fit = centroids_from_labeled({0: t0, 1: t1})
        assert fit is not None
        cents, radius = fit
        save_kit_ref(cfg_path, cents, radius, n_samples=(1, 1))
        save_centroids(match_level, cents, radius)  # bare — no kit meta
        save_centroids(run_path, cents * 0.5, radius)
        cfg = {"team_assignment": {"kit_centroids_path": str(cfg_path)}}
        got = resolve_kit_centroids_path(cfg, output_root=root, run_dir=run_dir)
        assert got == cfg_path
        cfg_empty = {"team_assignment": {"kit_centroids_path": ""}}
        got2 = resolve_kit_centroids_path(cfg_empty, output_root=root, run_dir=run_dir)
        assert got2 == match_level
        match_level.unlink()
        got3 = resolve_kit_centroids_path(cfg_empty, output_root=root, run_dir=run_dir)
        assert got3 == run_path


def test_tracklet_save_preserves_kit_meta():
    t0, t1 = [_feat(0.55, 0.12)], [_feat(0.1, 0.65)]
    fit = centroids_from_labeled({0: t0, 1: t1})
    assert fit is not None
    cents, radius = fit
    with tempfile.TemporaryDirectory() as td:
        src = Path(td) / "src.json"
        dst = Path(td) / "run" / "team_centroids.json"
        save_kit_ref(src, cents, radius, team_names=("A", "B"), n_samples=(2, 2))
        model = TrackletTeamModel()
        assert model.load(src)
        assert model.from_kit_ref
        model.save(dst)
        meta = load_kit_ref_meta(dst)
        assert meta.get("source") == "kit_label_dashboard"
        assert meta.get("team_names") == ["A", "B"]
        assert meta.get("n_samples") == [2, 2]


def test_kit_samples_bank_merge_and_backup():
    from src.perception.team_core import (
        backup_kit_ref,
        kit_samples_bank_path,
        load_kit_samples_bank,
        merge_kit_sample_rows,
        save_kit_samples_bank,
    )

    a = {"team": 0, "feat": _feat(0.7, 0.1).tolist(), "slot_key": "a", "frame_id": 1, "tag": "A"}
    b = {"team": 1, "feat": _feat(0.1, 0.7).tolist(), "slot_key": "b", "frame_id": 2, "tag": "B"}
    b2 = {"team": 1, "feat": _feat(0.12, 0.72).tolist(), "slot_key": "b", "frame_id": 3, "tag": "B2"}
    c = {"team": 0, "feat": _feat(0.65, 0.15).tolist(), "slot_key": "c", "frame_id": 4, "tag": "C"}
    merged = merge_kit_sample_rows([a, b], [b2, c])
    keys = {s["slot_key"] for s in merged}
    assert keys == {"a", "b", "c"}
    assert next(s for s in merged if s["slot_key"] == "b")["tag"] == "B2"
    with tempfile.TemporaryDirectory() as td:
        path = Path(td) / "team_centroids.json"
        fit = centroids_from_labeled(
            {
                0: [np.asarray(a["feat"], dtype=np.float32)],
                1: [np.asarray(b["feat"], dtype=np.float32)],
            }
        )
        assert fit is not None
        save_kit_ref(path, fit[0], fit[1], n_samples=(1, 1))
        save_kit_samples_bank(kit_samples_bank_path(path), [a, b])
        backup = backup_kit_ref(path)
        assert backup is not None and backup.is_file()
        rows = load_kit_samples_bank(kit_samples_bank_path(path))
        assert len(rows) == 2


if __name__ == "__main__":
    test_labeled_centroids()
    test_save_load_roundtrip()
    test_resolve_prefers_kit_path()
    test_tracklet_save_preserves_kit_meta()
    test_kit_samples_bank_merge_and_backup()
    print("ok")
