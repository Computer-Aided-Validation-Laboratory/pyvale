#===============================================================================
# pyvale: the python validation engine
# License: MIT
# Copyright (C) 2025 The Computer Aided Validation Team
#===============================================================================
from pathlib import Path

"""
DEVELOPER VERIFICATION MODULE
--------------------------------------------------------------------------------
This module contains developer utility functions used for verification testing
of the point sensor simulation toolbox in pyvale.

Specifically, this module contains constants used for verification testing.
"""

PROJECT_ROOT: Path = Path(__file__).resolve().parents[3]
GOLD_PATH: Path = PROJECT_ROOT / "tests" / "sensorsim" / "gold"
GOLD_SEED: int = 123
