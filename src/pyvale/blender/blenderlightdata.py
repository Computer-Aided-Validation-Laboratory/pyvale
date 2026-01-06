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

class LightType(Enum):
    POINT = 'POINT'
    SUN = 'SUN'
    SPOT = 'SPOT'
    AREA = 'AREA'

@dataclass(slots=True)
class LightData():
    pos_world: np.ndarray
    rot_world: Rotation
    energy: int # NOTE: In Watts
    type: LightType = LightType.POINT
    shadow_soft_size: float = 1.5

