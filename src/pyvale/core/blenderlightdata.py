"""
================================================================================
pyvale: the python validation engine
License: MIT
Copyright (C) 2024 The Computer Aided Validation Team
================================================================================
"""
from dataclasses import dataclass
from enum import Enum
import numpy as np
from scipy.spatial.transform import Rotation

class BlenderLightType(Enum):
    POINT = 'POINT'
    SUN = 'SUN'
    SPOT = 'SPOT'
    AREA = 'AREA'

@dataclass(slots=True)
class BlenderLightData():
    type: BlenderLightType
    pos_world: np.ndarray
    rot_world: Rotation
    energy: int

    def __post_init__(self) -> None:
        self.type = BlenderLightType.POINT

