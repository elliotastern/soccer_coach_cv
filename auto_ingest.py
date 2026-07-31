"""Lazy bridge to annotation/scripts/auto_ingest.py (loads only when imported symbols are used)."""
import importlib.util
from pathlib import Path
from typing import Any

_impl_mod: Any = None


def _ensure_impl():
    global _impl_mod
    if _impl_mod is not None:
        return _impl_mod
    impl_path = Path(__file__).resolve().parent / "annotation" / "scripts" / "auto_ingest.py"
    spec = importlib.util.spec_from_file_location("_annotation_auto_ingest_impl", impl_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load auto_ingest from {impl_path}")
    _impl_mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(_impl_mod)
    return _impl_mod


def __getattr__(name: str):
    if name in ("VideoHandler", "load_config", "get_default_config"):
        return getattr(_ensure_impl(), name)
    raise AttributeError(name)
