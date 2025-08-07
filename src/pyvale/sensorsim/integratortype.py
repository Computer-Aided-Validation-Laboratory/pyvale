# ==============================================================================
# pyvale: the python validation engine
# License: MIT
# Copyright (C) 2025 The Computer Aided Validation Team
# ==============================================================================

import enum


class EIntSpatialType(enum.Enum):
    """Enumeration specifying the type of spatial integrator to build. Used for
    spatial averaging for sensors.

    RECT1PT
        Rectangular 2D integrator splitting the area into 1 part.

    RECT4PT
        Rectangular 2D integrator splitting the area into 4 equal parts.

    RECT9PT
        Rectangular 2D integrator splitting the area into 9 equal parts.

    QUAD4PT
        Gaussian quadrature 2D integrator over 4 points.

    QUAD9PT
        Gaussian quadrature 2D integrator over 9 points.
    """

    RECT1PT = enum.auto()
    """Rectangular 2D integrator splitting the area into 1 part."""

    RECT4PT = enum.auto()
    """Rectangular 2D integrator splitting the area into 4 equal parts."""

    RECT9PT = enum.auto()
    """Rectangular 2D integrator splitting the area into 9 equal parts."""

    QUAD4PT = enum.auto()
    """Gaussian quadrature 2D integrator over 4 points."""

    QUAD9PT = enum.auto()
    """Gaussia quadrature 2D integrator over 9 points."""
