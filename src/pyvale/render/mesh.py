# ============================================================================
# pyvale: the python validation engine
# License: MIT
# Copyright (C) 2026 Sceptical Rabbit (Lloyd Fletcher)
# ============================================================================
"""Renderer independent surface mesh data."""

from dataclasses import dataclass
from enum import Enum

import numpy as np


class EElementType(Enum):
    """Surface element topologies supported by the 3D render API."""

    TRI3 = "tri3"
    TRI6 = "tri6"
    QUAD4 = "quad4"
    QUAD8 = "quad8"
    QUAD9 = "quad9"


_NODES_PER_ELEMENT = {
    EElementType.TRI3: 3,
    EElementType.TRI6: 6,
    EElementType.QUAD4: 4,
    EElementType.QUAD8: 8,
    EElementType.QUAD9: 9,
}


@dataclass(slots=True)
class Mesh3D:
    """A deformable surface mesh prepared for a three dimensional renderer.

    Parameters
    ----------
    element_type : EElementType
        Topology of every element in ``connectivity``.
    coords : np.ndarray
        World coordinates array with shape ``(num_nodes, 3)`` and dtype
        ``float64`` representing (X, Y, Z) coordinates of each node.
    connectivity : np.ndarray
        Zero based element connectivity table with shape
        ``(num_elements, nodes_per_element)`` and dtype ``uintp`` or ``int_``.
    shader : object
        Backend owned material or shader definition.
    displacements : np.ndarray or None, optional
        Nodal displacements array with shape
        ``(num_frames, num_nodes, 3)`` and dtype ``float64`` representing
        (dX, dY, dZ) displacements for each frame.
    """

    element_type: EElementType
    coords: np.ndarray
    connectivity: np.ndarray
    shader: object
    displacements: np.ndarray | None = None

    def __post_init__(self) -> None:
        """Convert array data to contiguous arrays with renderer dtypes."""
        self.coords = np.ascontiguousarray(self.coords, dtype=np.float64)
        self.connectivity = np.ascontiguousarray(
            self.connectivity,
            dtype=np.uintp,
        )
        if self.displacements is not None:
            self.displacements = np.ascontiguousarray(
                self.displacements,
                dtype=np.float64,
            )


__all__ = ["EElementType", "Mesh3D"]
