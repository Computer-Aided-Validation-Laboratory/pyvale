# ============================================================================
# pyvale: the python validation engine
# License: MIT
# Copyright (C) 2026 Sceptical Rabbit (Lloyd Fletcher)
# ============================================================================
"""Mesh and displacement containers for two-dimensional renderers."""

from dataclasses import dataclass

import numpy as np

from .mesh import EElementType


_NODES_PER_ELEMENT = {
    EElementType.TRI3: 3,
    EElementType.TRI6: 6,
    EElementType.QUAD4: 4,
    EElementType.QUAD8: 8,
    EElementType.QUAD9: 9,
}


@dataclass(slots=True)
class Mesh2D:
    """One Riley-ordered two-dimensional finite-element mesh.

    Parameters
    ----------
    element_type : EElementType
        Common topology of every mesh element.
    coords : numpy.ndarray
        Reference nodal coordinates with shape ``(nodes, 2)``.
    connectivity : numpy.ndarray
        Zero-based Riley-order node indices with shape
        ``(elements, nodes_per_element)``.
    """

    element_type: EElementType
    coords: np.ndarray
    connectivity: np.ndarray

    def __post_init__(self) -> None:
        """Normalise mesh arrays and validate topology and indices."""
        self.coords = np.ascontiguousarray(self.coords, dtype=np.float64)
        self.connectivity = np.ascontiguousarray(self.connectivity, dtype=np.intp)
        expected_nodes = _NODES_PER_ELEMENT[self.element_type]
        if self.coords.ndim != 2 or self.coords.shape[1] != 2:
            raise ValueError("coords must have shape (nodes, 2).")
        if (self.connectivity.ndim != 2
                or self.connectivity.shape[1] != expected_nodes
                or self.connectivity.shape[0] == 0):
            raise ValueError("connectivity does not match the element topology.")
        if (np.any(self.connectivity < 0)
                or np.any(self.connectivity >= self.coords.shape[0])):
            raise ValueError("connectivity contains an invalid node index.")


@dataclass(slots=True)
class DisplacementSeries2D:
    """Frame-major in-plane nodal displacement data.

    Parameters
    ----------
    values : numpy.ndarray
        Displacements with shape ``(frames, nodes, 2)``.
    """

    values: np.ndarray

    def __post_init__(self) -> None:
        """Normalise the displacement array to double precision."""
        self.values = np.ascontiguousarray(self.values, dtype=np.float64)
        if self.values.ndim != 3 or self.values.shape[2] != 2:
            raise ValueError("values must have shape (frames, nodes, 2).")


__all__ = ["DisplacementSeries2D", "Mesh2D"]
