'''
================================================================================
pyvale: the python validation engine
License: MIT
Copyright (C) 2024 The Digital Validation Team
================================================================================
'''
from dataclasses import dataclass
import numpy as np
from scipy.spatial.transform import Rotation
from pyvale.integratortype import EIntSpatialType

@dataclass(slots=True)
class SensorData:
    positions: np.ndarray | None = None
    sample_times: np.ndarray | None = None
    angles: tuple[Rotation,...] | None = None
    spatial_averager: EIntSpatialType | None = None
    spatial_dims: np.ndarray | None = None
