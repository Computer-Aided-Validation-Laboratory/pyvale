# ============================================================================
# pyvale: the python validation engine
# License: MIT
# Copyright (C) 2026 Sceptical Rabbit (Lloyd Fletcher)
# ============================================================================
"""Common three dimensional light data."""

from dataclasses import dataclass, field
from enum import Enum

import numpy as np


class ELightType(Enum):
    """Light geometries understood by compatible three dimensional backends."""

    POINT = "point"
    SUN = "sun"
    SPOT = "spot"
    AREA = "area"


@dataclass(slots=True)
class Light:
    """A physical light source for a renderer that supports lights.

    Parameters
    ----------
    light_type : ELightType
        Geometry of the light source.
    pos_world : np.ndarray
        World position array of the source with shape ``(3,)`` and dtype
        ``float64`` representing (X, Y, Z) coordinates.
    direction_world : np.ndarray
        World space direction vector array with shape ``(3,)`` and dtype
        ``float64`` representing (X, Y, Z) components. Relevant to sun and
        area lights.
    intensity : float
        Non negative backend independent light intensity.
    shadow_soft_size : float, optional
        Source radius used by backends that support soft shadows.
        Defaults to 1.5.
    """

    light_type: ELightType
    pos_world: np.ndarray
    direction_world: np.ndarray
    intensity: float
    shadow_soft_size: float = 1.5

    def __post_init__(self) -> None:
        """Convert position and direction to double precision arrays."""
        self.pos_world = np.asarray(self.pos_world, dtype=np.float64)
        self.direction_world = np.asarray(
            self.direction_world, dtype=np.float64
        )


__all__ = ["ELightType", "Light"]
