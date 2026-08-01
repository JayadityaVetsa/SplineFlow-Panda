"""Streamlit Community Cloud entry point.

The package uses a ``src`` layout. Adding it explicitly keeps the hosted dashboard
independent of whether Community Cloud installs the local project before launching the
script. Importing the dashboard executes the Streamlit page.
"""

import sys
from pathlib import Path

SOURCE = Path(__file__).resolve().parent / "src"
if str(SOURCE) not in sys.path:
    sys.path.insert(0, str(SOURCE))

import splineflow_panda.dashboard  # noqa: E402, F401
