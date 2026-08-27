# ==============================================================================
# pyvale: the python validation engine
# License: MIT
# Copyright (C) 2025 The Computer Aided Validation Team
# ==============================================================================
from enum import Enum, auto


class EDim(Enum):
    """Enumeration used to specify the number of spatial dimensions for a
    simulation. For mesh-based data this is used to determine the element type
    and distinguish between 4 node quads in 2D and 4 node tets in 3D. For point
    cloud data this determines if 2D or 3D Delaunay triangulation is used.
    """

    TWOD = 2
    THREED = 3


class EIntegrationMode(Enum):
    """Enumeration specifying the integration mode for spatial and temporal
    windows.

    AVERAGE:
        Weights are normalized so they sum to 1.0. Used for intensive physical
        quantities (temperature, strain, stress, pressure).
    ACCUMULATE:
        Weights represent physical metric measures (length, area, volume, time)
        so the sum equals the physical measure. Used for extensive quantities
        (total force, total heat energy, total power).
    """

    AVERAGE = auto()
    ACCUMULATE = auto()


