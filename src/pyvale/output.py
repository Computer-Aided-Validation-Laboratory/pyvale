# ==============================================================================
# pyvale: the python validation engine
# License: MIT
# Copyright (C) 2025 The Computer Aided Validation Team
# ==============================================================================
from dataclasses import dataclass
from pathlib import Path

@dataclass
class Outputs():
    base_dir = Path.home()