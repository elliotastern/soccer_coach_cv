"""Streamlit review dashboard entrypoint.

Usage:
    streamlit run apps/review_dashboard.py
"""
from pathlib import Path
import runpy

_app = Path(__file__).resolve().parents[1] / "src" / "review" / "app.py"
runpy.run_path(str(_app), run_name="__main__")
