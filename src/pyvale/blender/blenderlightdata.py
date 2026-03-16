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
    """
    Enumeration of available light types in Blender.
    
    Attributes
    ----------
    POINT : str
        Omnidirectional point light source
    SUN : str
        Directional sunlight (parallel rays)
    SPOT : str
        Cone-shaped spotlight
    AREA : str
        Area light with defined shape
    """
    
    POINT = 'POINT'
    SUN = 'SUN'
    SPOT = 'SPOT'
    AREA = 'AREA'

@dataclass(slots=True)
class LightData():
    """
    Configuration data for a light source in Blender.
    
    This dataclass stores the position, orientation, intensity, and type
    information for scene lighting.
    """
    
    pos_world: np.ndarray
    """Position of the light in world coordinates (x, y, z)"""
    
    rot_world: Rotation
    """Rotation of the light in world space (scipy Rotation object)"""
    
    energy: int
    """Light intensity in Watts"""
    
    type: LightType = LightType.POINT
    """Type of light source (default: POINT)"""
    
    shadow_soft_size: float = 1.5
    """Size of the light for soft shadow calculation (default: 1.5)"""

