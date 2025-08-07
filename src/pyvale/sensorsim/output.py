# ==============================================================================
# pyvale: the python validation engine
# License: MIT
# Copyright (C) 2025 The Computer Aided Validation Team
# ==============================================================================
from dataclasses import dataclass
from pathlib import Path

"""
NOTE: this module is a feature under developement.
"""

# TODO: remove this? but check blender tests

@dataclass(slots=True)
class Outputs():
    base_dir = Path.home()