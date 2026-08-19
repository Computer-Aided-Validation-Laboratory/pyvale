# ============================================================================
# pyvale: the python validation engine
# License: MIT
# Copyright (C) 2026 Sceptical Rabbit (Lloyd Fletcher)
# ============================================================================
"""Common three-dimensional light data."""

from dataclasses import dataclass
from enum import Enum

import numpy as np


class ELightType(Enum):
    """Light geometries understood by compatible backends."""

    POINT = "point"
    SUN = "sun"
    AREA = "area"


@dataclass(slots=True)
class Light:
    """A physical light source for a renderer that supports lights."""

    light_type: ELightType
    pos_world: np.ndarray
    direction_world: np.ndarray
    intensity: float

    def __post_init__(self) -> None:
        self.pos_world = np.asarray(self.pos_world, dtype=np.float64)
        self.direction_world = np.asarray(self.direction_world, dtype=np.float64)


__all__ = ["ELightType", "Light"]
