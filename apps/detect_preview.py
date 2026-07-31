"""Streamlit detection preview entrypoint.

Usage:
    streamlit run apps/detect_preview.py
"""
from pathlib import Path
import runpy
import sys

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

_app = _ROOT / "src" / "review" / "detect_viewer.py"
runpy.run_path(str(_app), run_name="__main__")
