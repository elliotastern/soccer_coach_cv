"""Fuse mode config: pitch_merge vs triangulate_3d."""
from __future__ import annotations

from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CFG = ROOT / "configs/default.yaml"
FUSE_MODES = ("pitch_merge", "triangulate_3d")
CAM_SETS = ("quad", "all")


def load_fuse_config(cfg_path: Path | None = None) -> dict:
    path = cfg_path or DEFAULT_CFG
    if not path.is_file():
        return default_fuse_config()
    cfg = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    fuse = cfg.get("fuse") or {}
    return {
        "mode": str(fuse.get("mode", "pitch_merge")),
        "ukf_enabled": bool(fuse.get("ukf_enabled", False)),
        "cams": str(fuse.get("cams", "all")),
        "reproj_max_px": dict(fuse.get("reproj_max_px") or {}),
        "fallback_pitch_merge": bool(fuse.get("fallback_pitch_merge", True)),
    }


def default_fuse_config() -> dict:
    return {
        "mode": "pitch_merge",
        "ukf_enabled": False,
        "cams": "all",
        "reproj_max_px": {},
        "fallback_pitch_merge": True,
    }


def fuse_cams_list(cfg: dict | None = None) -> list[str]:
    cfg = cfg or load_fuse_config()
    if cfg.get("cams") == "all":
        return ["P1", "P6", "P7", "P8", "P9", "P10", "P_Goal1", "P_Goal2"]
    return ["P7", "P8", "P9", "P10"]
