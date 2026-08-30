# ==============================================================================
# pyvale: the python validation engine
# License: MIT
# Copyright (C) 2026 Sceptical Rabbit (Lloyd Fletcher)
# ==============================================================================
"""Spatial layout and multi object arrangement helpers for render scenes."""

from collections.abc import Sequence

import numpy as np
from scipy.spatial.transform import Rotation

from .mesh import Mesh3D
from .meshtools import (
    mesh_bounds,
    mesh_center,
    mesh_center_at,
    mesh_rotate,
    mesh_translate,
)


def scene_bounds(
    meshes: Sequence[Mesh3D],
) -> tuple[np.ndarray, np.ndarray]:
    """Calculate the bounding box enclosing all meshes in a scene.

    Parameters
    ----------
    meshes : Sequence[Mesh3D]
        Collection of meshes to measure.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        Tuple of ``(coord_min, coord_max)`` arrays, each with shape ``(3,)``
        and dtype ``float64`` representing (X, Y, Z) minimum and maximum
        extents of the enclosing bounding box.
    """
    if not meshes:
        return np.zeros(3), np.zeros(3)

    lowers, uppers = zip(
        *(mesh_bounds(mesh) for mesh in meshes),
        strict=True,
    )

    return np.min(np.array(lowers), axis=0), np.max(np.array(uppers), axis=0)


def scene_center(meshes: Sequence[Mesh3D]) -> np.ndarray:
    """Calculate the midpoint of the scene bounding box.

    Parameters
    ----------
    meshes : Sequence[Mesh3D]
        Collection of meshes.

    Returns
    -------
    np.ndarray
        Midpoint coordinates array with shape ``(3,)`` and dtype
        ``float64`` representing (X, Y, Z) center of the scene.
    """
    lower, upper = scene_bounds(meshes)
    return 0.5 * (lower + upper)


def scene_translate(
    meshes: Sequence[Mesh3D],
    translation: np.ndarray,
) -> list[Mesh3D]:
    """Translate every mesh in a sequence by the same offset vector.

    Parameters
    ----------
    meshes : Sequence[Mesh3D]
        Collection of meshes to translate.
    translation : np.ndarray
        Translation offset array with shape ``(3,)`` or ``(2,)`` and dtype
        ``float64`` representing (dX, dY, dZ) offsets.

    Returns
    -------
    list[Mesh3D]
        New list of translated meshes.
    """
    return [mesh_translate(mesh, translation) for mesh in meshes]


def scene_rotate(
    meshes: Sequence[Mesh3D],
    rotation: Rotation,
    pivot: np.ndarray | None = None,
) -> list[Mesh3D]:
    """Rotate all meshes around a shared pivot (default: scene center).

    Parameters
    ----------
    meshes : Sequence[Mesh3D]
        Collection of meshes to rotate.
    rotation : scipy.spatial.transform.Rotation
        Spatial rotation to apply to all meshes.
    pivot : np.ndarray or None, optional
        Pivot center point array with shape ``(3,)`` and dtype ``float64``.
        If ``None``, defaults to the scene center.

    Returns
    -------
    list[Mesh3D]
        New list of rotated meshes.
    """
    p_vec = scene_center(meshes) if pivot is None else pivot
    return [mesh_rotate(mesh, rotation, pivot=p_vec) for mesh in meshes]


def scene_arrange_points(
    meshes: Sequence[Mesh3D],
    positions: np.ndarray,
) -> list[Mesh3D]:
    """Move each mesh so its bounding box center lies at the matching target.

    Parameters
    ----------
    meshes : Sequence[Mesh3D]
        Collection of meshes to arrange.
    positions : np.ndarray
        Array of target center positions with shape ``(num_meshes, 3)`` and
        dtype ``float64`` representing (X, Y, Z) coordinates for each mesh.

    Returns
    -------
    list[Mesh3D]
        New list of repositioned meshes.

    Raises
    ------
    ValueError
        If the number of meshes does not match the number of positions.
    """
    if len(meshes) != len(positions):
        raise ValueError(
            f"Number of meshes ({len(meshes)}) must match number of "
            f"positions ({len(positions)})."
        )

    return [
        mesh_center_at(mesh, position)
        for mesh, position in zip(meshes, positions, strict=True)
    ]


def scene_arrange_line(
    meshes: Sequence[Mesh3D],
    axis: int = 0,
    spacing: float = 0.0,
) -> list[Mesh3D]:
    """Arrange meshes in a line with a constant gap between bounding boxes.

    Parameters
    ----------
    meshes : Sequence[Mesh3D]
        Collection of meshes to arrange.
    axis : int, optional
        Spatial axis index along which to align meshes (0 for X, 1 for Y,
        2 for Z). Defaults to 0.
    spacing : float, optional
        Clearance distance between successive bounding boxes. Defaults to 0.0.

    Returns
    -------
    list[Mesh3D]
        New list of arranged meshes.
    """
    if not meshes:
        return []

    arranged: list[Mesh3D] = []
    current_edge: float | None = None

    for mesh in meshes:
        low, high = mesh_bounds(mesh)
        if current_edge is None:
            arranged.append(mesh)
            current_edge = high[axis]
        else:
            shift = (current_edge + spacing) - low[axis]
            delta = np.zeros_like(low)
            delta[axis] = shift
            moved = mesh_translate(mesh, delta)
            arranged.append(moved)
            _, moved_high = mesh_bounds(moved)
            current_edge = moved_high[axis]

    return arranged


def scene_arrange_grid(
    meshes: Sequence[Mesh3D],
    columns: int = 3,
    spacing: np.ndarray = np.array((0.0, 0.0)),
    plane: str = "xy",
    center: bool = True,
) -> list[Mesh3D]:
    """Arrange meshes in a 2D planar grid with defined gaps between boxes.

    Parameters
    ----------
    meshes : Sequence[Mesh3D]
        Collection of meshes to arrange in a grid.
    columns : int, optional
        Number of grid columns. Defaults to 3.
    spacing : np.ndarray, optional
        Gaps between bounding boxes with shape ``(2,)`` and dtype ``float64``
        representing ``(gap_u, gap_v)``. Defaults to ``(0.0, 0.0)``.
    plane : str, optional
        Grid arrangement plane (``"xy"``, ``"xz"``, or ``"yz"``). Defaults
        to ``"xy"``.
    center : bool, optional
        Whether to center the entire grid at the origin. Defaults to ``True``.

    Returns
    -------
    list[Mesh3D]
        New list of grid arranged meshes.

    Raises
    ------
    ValueError
        If columns is not positive or plane is unsupported.
    """
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

    for ii, mesh in enumerate(meshes):
        col = ii % columns
        row = ii // columns
        low, high = mesh_bounds(mesh)
        width = high[u_axis] - low[u_axis]
        height = high[v_axis] - low[v_axis]
        col_widths[col] = max(col_widths[col], width)
        row_heights[row] = max(row_heights[row], height)

    col_offsets = np.zeros(columns, dtype=np.float64)
    for cc in range(1, columns):
        col_offsets[cc] = col_offsets[cc - 1] + col_widths[cc - 1] + gap_u

    row_offsets = np.zeros(rows, dtype=np.float64)
    for rr in range(1, rows):
        row_offsets[rr] = row_offsets[rr - 1] + row_heights[rr - 1] + gap_v

    arranged: list[Mesh3D] = []
    for ii, mesh in enumerate(meshes):
        col = ii % columns
        row = ii // columns
        low, high = mesh_bounds(mesh)
        width = high[u_axis] - low[u_axis]
        height = high[v_axis] - low[v_axis]

        target_u = col_offsets[col] + 0.5 * (col_widths[col] - width)
        target_v = row_offsets[row] + 0.5 * (row_heights[row] - height)

        delta = np.zeros_like(low)
        delta[u_axis] = target_u - low[u_axis]
        delta[v_axis] = target_v - low[v_axis]
        arranged.append(mesh_translate(mesh, delta))

    if center and arranged:
        sc = scene_center(arranged)
        arranged = scene_translate(arranged, -sc)

    return arranged


def scene_arrange_circle(
    meshes: Sequence[Mesh3D],
    radius: float = 100.0,
    plane: str = "xy",
    center: np.ndarray = np.array((0.0, 0.0, 0.0)),
) -> list[Mesh3D]:
    """Place mesh centers evenly spaced around a circle.

    Parameters
    ----------
    meshes : Sequence[Mesh3D]
        Collection of meshes to arrange in a circle.
    radius : float, optional
        Radius of the circle. Defaults to 100.0.
    plane : str, optional
        Circle plane (``"xy"``, ``"xz"``, or ``"yz"``). Defaults to ``"xy"``.
    center : np.ndarray, optional
        Center coordinates of the circle with shape ``(3,)`` and dtype
        ``float64``. Defaults to ``(0.0, 0.0, 0.0)``.

    Returns
    -------
    list[Mesh3D]
        New list of circularly arranged meshes.

    Raises
    ------
    ValueError
        If plane is unsupported.
    """
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

    c_vec = np.asarray(center, dtype=np.float64)[:3]
    n = len(meshes)
    angles = np.linspace(0.0, 2.0 * np.pi, n, endpoint=False)

    arranged: list[Mesh3D] = []
    for mesh, angle in zip(meshes, angles, strict=True):
        pos = c_vec.copy()
        pos[u_axis] += radius * np.cos(angle)
        pos[v_axis] += radius * np.sin(angle)
        arranged.append(mesh_center_at(mesh, pos))

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
