# ==============================================================================
# pyvale: the python validation engine
# License: MIT
# Copyright (C) 2026 Sceptical Rabbit (Lloyd Fletcher)
# ==============================================================================
"""Spatial layout and multi-object arrangement helpers for render scenes."""

from collections.abc import Sequence
from typing import TypeVar

import numpy as np
from scipy.spatial.transform import Rotation

from .mesh import Mesh2D, Mesh3D
from .meshtools import (
    mesh_bounds,
    mesh_center,
    mesh_center_at,
    mesh_rotate,
    mesh_translate,
)

MeshT = TypeVar("MeshT", Mesh2D, Mesh3D)


def scene_bounds(
    meshes: Sequence[Mesh2D | Mesh3D],
) -> tuple[np.ndarray, np.ndarray]:
    """Calculate the bounding box enclosing all meshes in a scene.

    Parameters
    ----------
    meshes : Sequence[Mesh2D or Mesh3D]
        Collection of meshes to measure.

    Returns
    -------
    tuple[numpy.ndarray, numpy.ndarray]
        Lower ``(min)`` and upper ``(max)`` spatial extents.
    """
    if not meshes:
        return np.zeros(3), np.zeros(3)

    lowers, uppers = zip(*(mesh_bounds(m) for m in meshes), strict=True)

    return np.min(np.array(lowers), axis=0), np.max(np.array(uppers), axis=0)


def scene_center(meshes: Sequence[Mesh2D | Mesh3D]) -> np.ndarray:
    """Calculate the midpoint of the scene bounding box.

    Parameters
    ----------
    meshes : Sequence[Mesh2D or Mesh3D]
        Collection of meshes.

    Returns
    -------
    numpy.ndarray
        Center coordinates of the enclosing bounding box.
    """
    lower, upper = scene_bounds(meshes)
    return 0.5 * (lower + upper)


def scene_translate(
    meshes: Sequence[MeshT],
    translation: Sequence[float],
) -> list[MeshT]:
    """Translate every mesh in a sequence by the same offset vector."""
    return [mesh_translate(m, translation) for m in meshes]


def scene_rotate(
    meshes: Sequence[MeshT],
    rotation: Rotation,
    pivot: Sequence[float] | None = None,
) -> list[MeshT]:
    """Rotate all meshes around a shared pivot (default: scene center)."""
    p_vec = scene_center(meshes) if pivot is None else pivot
    return [mesh_rotate(m, rotation, pivot=p_vec) for m in meshes]


def scene_arrange_points(
    meshes: Sequence[MeshT],
    positions: Sequence[Sequence[float]],
) -> list[MeshT]:
    """Move each mesh so its bounding box center lies at the matching target."""

    if len(meshes) != len(positions):
        raise ValueError(
            f"Number of meshes ({len(meshes)}) must match number of "
            f"positions ({len(positions)})."
        )

    return [
        mesh_center_at(m, pos)
        for m, pos in zip(meshes, positions, strict=True)
    ]


def scene_arrange_line(
    meshes: Sequence[MeshT],
    axis: int = 0,
    spacing: float = 0.0,
) -> list[MeshT]:
    """Arrange meshes in a line with a constant gap between bounding boxes."""
    if not meshes:
        return []

    arranged: list[MeshT] = []
    current_edge: float | None = None

    for m in meshes:
        low, high = mesh_bounds(m)
        if current_edge is None:
            arranged.append(m)
            current_edge = high[axis]
        else:
            shift = (current_edge + spacing) - low[axis]
            delta = np.zeros_like(low)
            delta[axis] = shift
            moved = mesh_translate(m, delta)
            arranged.append(moved)
            _, moved_high = mesh_bounds(moved)
            current_edge = moved_high[axis]

    return arranged


def scene_arrange_grid(
    meshes: Sequence[MeshT],
    columns: int = 3,
    spacing: Sequence[float] = (0.0, 0.0),
    plane: str = "xy",
    center: bool = True,
) -> list[MeshT]:
    """Arrange meshes in a 2D planar grid with defined gaps between boxes."""

    if not meshes:
        return []

    if columns <= 0:
        raise ValueError("columns must be a positive integer.")

    plane_lower = plane.lower()
    if plane_lower == "xy":
        u_axis, v_axis = 0, 1
    elif plane_lower == "xz":
        u_axis, v_axis = 0, 2
    elif plane_lower == "yz":
        u_axis, v_axis = 1, 2
    else:
        raise ValueError(f"Unsupported plane '{plane}'. Choose xy, xz, or yz.")

    gap_u = float(spacing[0])
    gap_v = float(spacing[1]) if len(spacing) > 1 else gap_u

    rows = (len(meshes) + columns - 1) // columns
    col_widths = np.zeros(columns, dtype=np.float64)
    row_heights = np.zeros(rows, dtype=np.float64)

    for i, m in enumerate(meshes):
        col = i % columns
        row = i // columns
        low, high = mesh_bounds(m)
        w = high[u_axis] - low[u_axis]
        h = high[v_axis] - low[v_axis]
        col_widths[col] = max(col_widths[col], w)
        row_heights[row] = max(row_heights[row], h)

    col_offsets = np.zeros(columns, dtype=np.float64)
    for c in range(1, columns):
        col_offsets[c] = col_offsets[c - 1] + col_widths[c - 1] + gap_u

    row_offsets = np.zeros(rows, dtype=np.float64)
    for r in range(1, rows):
        row_offsets[r] = row_offsets[r - 1] + row_heights[r - 1] + gap_v

    arranged: list[MeshT] = []
    for i, m in enumerate(meshes):
        col = i % columns
        row = i // columns
        low, high = mesh_bounds(m)
        w = high[u_axis] - low[u_axis]
        h = high[v_axis] - low[v_axis]

        target_u = col_offsets[col] + 0.5 * (col_widths[col] - w)
        target_v = row_offsets[row] + 0.5 * (row_heights[row] - h)

        delta = np.zeros_like(low)
        delta[u_axis] = target_u - low[u_axis]
        delta[v_axis] = target_v - low[v_axis]
        arranged.append(mesh_translate(m, delta))

    if center and arranged:
        sc = scene_center(arranged)
        arranged = scene_translate(arranged, -sc)

    return arranged


def scene_arrange_circle(
    meshes: Sequence[MeshT],
    radius: float = 100.0,
    plane: str = "xy",
    center: Sequence[float] = (0.0, 0.0, 0.0),
) -> list[MeshT]:
    """Place mesh centers evenly spaced around a circle."""
    if not meshes:
        return []

    plane_lower = plane.lower()
    if plane_lower == "xy":
        u_axis, v_axis = 0, 1
    elif plane_lower == "xz":
        u_axis, v_axis = 0, 2
    elif plane_lower == "yz":
        u_axis, v_axis = 1, 2
    else:
        raise ValueError(f"Unsupported plane '{plane}'. Choose xy, xz, or yz.")

    c_vec = np.asarray(center, dtype=np.float64)
    dims = 2 if isinstance(meshes[0], Mesh2D) else 3
    if c_vec.size < dims:
        c_vec = np.pad(c_vec, (0, dims - c_vec.size))
    else:
        c_vec = c_vec[:dims]

    n = len(meshes)
    angles = np.linspace(0.0, 2.0 * np.pi, n, endpoint=False)

    arranged: list[MeshT] = []
    for m, angle in zip(meshes, angles, strict=True):
        pos = c_vec.copy()
        pos[u_axis] += radius * np.cos(angle)
        pos[v_axis] += radius * np.sin(angle)
        arranged.append(mesh_center_at(m, pos))

    return arranged


__all__ = [
    "scene_arrange_circle",
    "scene_arrange_grid",
    "scene_arrange_line",
    "scene_arrange_points",
    "scene_bounds",
    "scene_center",
    "scene_rotate",
    "scene_translate",
]
