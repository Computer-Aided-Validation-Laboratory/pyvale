# ==============================================================================
# pyvale: the python validation engine
# License: MIT
# Copyright (C) 2025 The Computer Aided Validation Team
# ==============================================================================
from enum import Enum

class EDim(Enum):
    """Enumeration used to specify the number of spatial dimensions for a 
    simulation. For mesh-based data this is used to determine the element type
    and distinguish between 4 node quads in 2D and 4 node tets in 3D. For point
    cloud data this determines if 2D or 3D Delaunay triangulation is used. 
    """
    TWOD = 2
    THREED = 3

