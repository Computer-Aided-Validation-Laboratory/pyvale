# ============================================================================
# pyvale: the python validation engine
# License: MIT
# Copyright (C) 2026 Sceptical Rabbit (Lloyd Fletcher)
# ============================================================================
"""Conversion from simulation results to renderer surface meshes."""

from collections.abc import Sequence

import numpy as np

from pyvale.dataio.meshconv import (
    enforce_mesh_convention,
    extract_surf_mesh,
    is_mesh_2d,
)
from pyvale.dataio.simdata import SimData

from .mesh import EElementType, Mesh2D, Mesh3D


def mesh3d_from_simdata(
    sim_data: SimData,
    shader: object,
    displacement_keys: Sequence[str] | None = None,
) -> Mesh3D:
    """Build one surface :class:`Mesh3D` from simulation data.

    Parameters
    ----------
    sim_data : pyvale.dataio.SimData
        Simulation data containing coordinates, connectivity, and optionally
        nodal displacement fields.
    shader : object
        Backend-owned material or shader definition for the mesh.
    displacement_keys : Sequence[str] or None, optional
        Names of the three displacement components. ``None`` omits motion.
    Returns
    -------
    Mesh3D
        Renderer-independent surface mesh data.

    Raises
    ------
    ValueError
        If required mesh data is absent or does not contain exactly one
        supported connectivity table.

    Notes
    -----
    Volumetric simulation meshes are always skinned before conversion. A
    :class:`Mesh3D` has one topology, so split simulations with several
    connectivity tables into separate render meshes before conversion.
    """
    prepared = enforce_mesh_convention(sim_data)

    if not is_mesh_2d(prepared):
        prepared = extract_surf_mesh(prepared)

    coords, connectivity = _single_surface_table(prepared)
    element_type = _element_type_from_nodes(connectivity.shape[1])
    displacements = _displacements_from_simdata(prepared, displacement_keys)
    return Mesh3D(
        element_type=element_type,
        coords=_coords3d(coords),
        connectivity=np.ascontiguousarray(connectivity, dtype=np.uintp),
        shader=shader,
        displacements=displacements,
    )


def mesh2d_from_simdata(
    sim_data: SimData,
    displacement_keys: Sequence[str] | None = None,
) -> Mesh2D:
    """Build one XY-planar :class:`Mesh2D` from simulation data.

    The input must already be a two-dimensional surface mesh in the XY plane.
    Coordinates with a Z column are accepted only when every Z value is zero
    within the shared mesh-convention tolerance.
    """
    prepared = enforce_mesh_convention(sim_data)
    if not is_mesh_2d(prepared):
        raise ValueError("mesh2d_from_simdata requires a surface mesh.")

    coords, connectivity = _single_surface_table(prepared)
    coords2d = _coords2d_xy(coords)
    element_type = _element_type_from_nodes(connectivity.shape[1])
    displacement = _displacements_from_simdata(prepared, displacement_keys)

    if displacement is not None:
        displacement = displacement[:, :, :2]

    return Mesh2D(
        element_type=element_type,
        coords=coords2d,
        connectivity=np.ascontiguousarray(connectivity, dtype=np.intp),
        displacement=displacement,
    )


def _single_surface_table(sim_data: SimData) -> tuple[np.ndarray, np.ndarray]:
    """Return the one required surface connectivity table and coordinates."""
    if sim_data.coords is None or sim_data.connect is None:
        raise ValueError("SimData must provide coordinates and connectivity.")

    if len(sim_data.connect) != 1:
        raise ValueError("SimData must have exactly one connectivity table.")

    return sim_data.coords, next(iter(sim_data.connect.values()))


def _coords3d(coords: np.ndarray) -> np.ndarray:
    """Return finite render coordinates padded to three dimensions."""
    coords_out = np.ascontiguousarray(coords, dtype=np.float64)
    if coords_out.ndim != 2 or coords_out.shape[1] not in (2, 3):
        raise ValueError("SimData coordinates must have two or three columns.")

    if coords_out.shape[1] == 2:
        coords_out = np.pad(coords_out, ((0, 0), (0, 1)))

    return coords_out


def _coords2d_xy(coords: np.ndarray) -> np.ndarray:
    """Return XY coordinates, rejecting non-planar input."""
    coords_out = np.ascontiguousarray(coords, dtype=np.float64)
    if coords_out.ndim != 2 or coords_out.shape[1] not in (2, 3):
        raise ValueError("SimData coordinates must have two or three columns.")

    if coords_out.shape[1] == 3 and not np.allclose(
        coords_out[:, 2],
        0.0,
        atol=1.0e-12,
    ):
        raise ValueError("mesh2d_from_simdata only supports the XY plane.")

    return np.ascontiguousarray(coords_out[:, :2], dtype=np.float64)


def _element_type_from_nodes(nodes_per_element: int) -> EElementType:
    """Map a connectivity width to the corresponding render topology.

    Parameters
    ----------
    nodes_per_element : int
        Number of node indices in each connectivity row.

    Returns
    -------
    EElementType
        Renderer topology matching the connectivity width.

    Raises
    ------
    ValueError
        If the width has no matching supported topology.
    """
    element_types = {
        3: EElementType.TRI3,
        4: EElementType.QUAD4,
        6: EElementType.TRI6,
        8: EElementType.QUAD8,
        9: EElementType.QUAD9,
    }
    try:
        return element_types[nodes_per_element]
    except KeyError as error:
        raise ValueError(
            "No render element type for "
            f"{nodes_per_element} nodes per element.",
        ) from error


def _displacements_from_simdata(
    sim_data: SimData,
    displacement_keys: Sequence[str] | None,
) -> np.ndarray | None:
    """Extract three nodal displacement fields into renderer array order.

    Parameters
    ----------
    sim_data : pyvale.dataio.SimData
        Simulation data containing nodal variables.
    displacement_keys : Sequence[str] or None
        Names of the x, y, and z displacement variables.

    Returns
    -------
    numpy.ndarray or None
        Displacements with shape ``(frames, nodes, 3)``, or ``None`` when no
        keys are requested.

    Raises
    ------
    ValueError
        If the fields are missing or do not have ``(nodes, frames)`` shape.
    """
    if displacement_keys is None:
        return None
    if sim_data.node_vars is None or len(displacement_keys) not in (2, 3):
        raise ValueError("Two or three displacement keys are required.")
    try:
        fields = [
            np.asarray(sim_data.node_vars[key], dtype=np.float64)
            for key in displacement_keys
        ]
    except KeyError as error:
        raise ValueError(
            f"Missing displacement field {error.args[0]!r}.",
        ) from error
    if any(field.ndim != 2 for field in fields):
        raise ValueError("Displacement fields must have shape (nodes, frames).")
    displacements = np.stack(fields, axis=2).transpose(1, 0, 2)
    if displacements.shape[2] == 2:
        displacements = np.pad(displacements, ((0, 0), (0, 0), (0, 1)))
    return np.ascontiguousarray(displacements)


__all__ = [
    "mesh2d_from_simdata",
    "mesh3d_from_simdata",
]
