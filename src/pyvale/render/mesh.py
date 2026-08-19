# ============================================================================
# pyvale: the python validation engine
# License: MIT
# Copyright (C) 2026 Sceptical Rabbit (Lloyd Fletcher)
# ============================================================================
"""Renderer-independent surface mesh data."""

from dataclasses import dataclass
from enum import Enum

import numpy as np


class EElementType(Enum):
    """Surface element topologies shared by the 3D render API."""

    TRI3 = "tri3"
    TRI6 = "tri6"
    QUAD4 = "quad4"
    QUAD8 = "quad8"
    QUAD9 = "quad9"


@dataclass(slots=True)
class Mesh:
    """A deformable surface mesh prepared for a renderer.

    Coordinates have shape ``(node_count, 3)`` and displacements, when
    present, have shape ``(frame_count, node_count, 3)``.
    """

    element_type: EElementType
    coords: np.ndarray
    connectivity: np.ndarray
    shader: object
    displacements: np.ndarray | None = None

    def __post_init__(self) -> None:
        self.coords = np.ascontiguousarray(self.coords, dtype=np.float64)
        self.connectivity = np.ascontiguousarray(self.connectivity, dtype=np.uintp)
        if self.displacements is not None:
            self.displacements = np.ascontiguousarray(
                self.displacements, dtype=np.float64,
            )


__all__ = ["EElementType", "Mesh"]
