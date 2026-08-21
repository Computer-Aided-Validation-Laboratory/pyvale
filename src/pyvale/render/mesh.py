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
    """A deformable surface mesh prepared for a three-dimensional renderer.

    Parameters
    ----------
    element_type : EElementType
        Topology of every element in ``connectivity``.
    coords : numpy.ndarray
        World coordinates with shape ``(node_count, 3)``.
    connectivity : numpy.ndarray
        Zero-based node indices with shape
        ``(element_count, nodes_per_element)``.
    shader : object
        Backend-owned material or shader definition.
    displacements : numpy.ndarray or None, optional
        Nodal displacements with shape ``(frame_count, node_count, 3)``.
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
                self.displacements, dtype=np.float64,
            )


@dataclass(slots=True)
class Mesh2D:
    """Riley/VTK convention two-dimensional rendering finite element mesh.

    Parameters
    ----------
    element_type : EElementType
        Common topology of every mesh element.
    coords : numpy.ndarray
        Reference nodal coordinates with shape ``(nodes, 2)``.
    connectivity : numpy.ndarray
        Zero-based Riley-order node indices with shape
        ``(elements, nodes_per_element)``.
    displacement : numpy.ndarray or None, optional
        Nodal displacements with shape ``(frame_count, node_count, 2)``.
        ``None`` creates one undeformed frame.
    """

    element_type: EElementType
    coords: np.ndarray
    connectivity: np.ndarray
    displacement: np.ndarray | None = None

    def __post_init__(self) -> None:
        """Normalise mesh arrays and validate topology and indices."""

        self.coords = np.ascontiguousarray(self.coords, dtype=np.float64)
        self.connectivity = np.ascontiguousarray(
            self.connectivity,
            dtype=np.intp,
        )

        if self.displacement is None:
            self.displacement = np.zeros((1, self.coords.shape[0], 2))
        else:
            self.displacement = np.ascontiguousarray(
                self.displacement,
                dtype=np.float64,
            )

        if self.coords.ndim != 2 or self.coords.shape[1] != 2:
            raise ValueError("coords must have shape (nodes, 2).")

        expected_nodes = _NODES_PER_ELEMENT[self.element_type]
        if (
            self.connectivity.ndim != 2
            or self.connectivity.shape[1] != expected_nodes
            or self.connectivity.shape[0] == 0
        ):
            raise ValueError(
                "connectivity does not match the element topology.",
            )

        if (
            np.any(self.connectivity < 0)
            or np.any(self.connectivity >= self.coords.shape[0])
        ):
            raise ValueError("connectivity contains an invalid node index.")

        if self.displacement.ndim != 3 or self.displacement.shape[2] != 2:
            raise ValueError(
                "Displacement must have shape (frames, nodes, 2).",
            )

        if self.displacement.shape[1] != self.coords.shape[0]:
            raise ValueError("Displacement nodes must match mesh nodes.")


__all__ = ["EElementType", "Mesh3D", "Mesh2D"]
