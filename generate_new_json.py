#!/usr/bin/env python3
"""Compatibility wrapper for fontdata_example/generate_new_json.py."""

import runpy
from pathlib import Path


SCRIPT = Path(__file__).resolve().parent / "fontdata_example" / "generate_new_json.py"


if __name__ == "__main__":
    runpy.run_path(str(SCRIPT), run_name="__main__")
