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


class EDifferentialMode(Enum):
    """Enumeration specifying the differential sensor reduction mode.

    DIFFERENCE:
        Scalar difference M = M_B - M_A (e.g. Delta T, Delta P, Delta u).
    STRAIN:
        Tensile engineering strain:
        eps = ((u_B - u_A) . e_AB) / L_0
        where L_0 is the initial undeformed gauge length.
    RATIO:
        Ratio of measurements M = M_B / M_A.
    CUSTOM:
        User-defined reduction callable func(meas_a, meas_b) -> np.ndarray.
    """

    DIFFERENCE = auto()
    STRAIN = auto()
    RATIO = auto()
    CUSTOM = auto()


class ERayMode(Enum):
    """Enumeration specifying the measurement mode for ray sensors.

    DISTANCE:
        Measures standoff distance d = ||x_hit - x_0|| from ray origin to the
        deformed surface intersection.
    SURFACE_FIELD:
        Samples an underlying physical field (temperature, radiance, pressure)
        at the dynamic surface intersection point x_hit.
    LINE_OF_SIGHT:
        Integrates a field along the ray path segment from origin to surface.
    """

    DISTANCE = auto()
    SURFACE_FIELD = auto()
    LINE_OF_SIGHT = auto()



