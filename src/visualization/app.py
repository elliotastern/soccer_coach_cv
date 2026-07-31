# Compatibility shim for Streamlit / legacy path.
# Prefer: streamlit run apps/review_dashboard.py
from pathlib import Path
import runpy

_review_app = Path(__file__).resolve().parents[1] / "review" / "app.py"
runpy.run_path(str(_review_app), run_name="__main__")
