"""
================================================================================
pyvale: the python validation engine
License: MIT
Copyright (C) 2024 The Computer Aided Validation Team
================================================================================
"""
from dataclasses import dataclass

@dataclass(slots=True)
class BlenderMaterialData():
    # TODO: Add other material properties here
    roughness: float = 1.0
    metallic: float = 0.0
    interpolant: int = 'Cubic'