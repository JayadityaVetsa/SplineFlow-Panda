"""Streamlit Community Cloud entry point.

The package uses a ``src`` layout. Adding it explicitly keeps the hosted dashboard
independent of whether Community Cloud installs the local project before launching the
script. Importing the dashboard executes the Streamlit page.
"""

import runpy
import sys
from pathlib import Path

SOURCE = Path(__file__).resolve().parent / "src"
if str(SOURCE) not in sys.path:
    sys.path.insert(0, str(SOURCE))

runpy.run_path(str(SOURCE / "splineflow_panda" / "dashboard.py"), run_name="__main__")
