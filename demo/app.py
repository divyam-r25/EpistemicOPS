"""
EpistemicOps Demo — Redirect
==============================
The main dashboard is now in the root app.py.
This file is kept for backwards compatibility.

Usage:
    python app.py  (from project root)
"""
import sys
from pathlib import Path

# Redirect to root app.py
sys.path.insert(0, str(Path(__file__).parent.parent))

if __name__ == "__main__":
    from app import build_ui
    app = build_ui()
    app.launch(share=False)
