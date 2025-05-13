# ==============================================================================
# pyvale: the python validation engine
# License: MIT
# Copyright (C) 2025 The Computer Aided Validation Team
# ==============================================================================
from dataclasses import dataclass
from enum import Enum
import numpy as np
from scipy.spatial.transform import Rotation

#TODO: docstrings

class BlenderLightType(Enum):
    POINT = 'POINT'
    SUN = 'SUN'
    SPOT = 'SPOT'
    AREA = 'AREA'

@dataclass(slots=True)
class BlenderLightData():
    pos_world: np.ndarray
    rot_world: Rotation
    energy: int # NOTE: In Watts
    type: BlenderLightType = BlenderLightType.POINT
    shadow_soft_size: float = 1.5

