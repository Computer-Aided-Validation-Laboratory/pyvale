"""
================================================================================
pyvale: the python validation engine
License: MIT
Copyright (C) 2024 The Computer Aided Validation Team
================================================================================
"""
from dataclasses import dataclass

@dataclass
class BlenderMaterialData():
    # TODO: Add other material properties here
    roughness: float | None = None
    metallic: float | None = None
    interpolant = 'Cubic'

    def __post_init__(self) -> None:
        self.roughness = 1
        self.metallic = 0