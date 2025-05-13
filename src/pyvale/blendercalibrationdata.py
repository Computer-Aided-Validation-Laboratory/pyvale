# ==============================================================================
# pyvale: the python validation engine
# License: MIT
# Copyright (C) 2025 The Computer Aided Validation Team
# ==============================================================================
from dataclasses import dataclass

#TODO: doctsrings

@dataclass(slots=True)
class CalibrationData:
    angle_lims: tuple = (-10, 10)
    angle_step: int = 5
    plunge_lims: tuple = (-5, 5)
    plunge_step: int = 5