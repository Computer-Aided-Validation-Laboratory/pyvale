# ============================================================================
# pyvale: the python validation engine
# License: MIT
# Copyright (C) 2026 Sceptical Rabbit (Lloyd Fletcher)
# ============================================================================
"""Conversion from simulation results to renderer meshes."""

from collections.abc import Sequence

import numpy as np

from pyvale.dataio.meshtools import enforce_mesh_convention, extract_surf_mesh
from pyvale.dataio.simdata import SimData

from .mesh import EElementType, Mesh3D


def mesh_from_simdata(
    sim_data: SimData,
    shader: object,
    displacement_keys: Sequence[str] | None = None,
    extract_surface: bool = False,
) -> Mesh3D:
    """Build one :class:`Mesh3D` from convention-normalised simulation data.

    Parameters
    ----------
    sim_data : pyvale.dataio.SimData
        Simulation data containing coordinates, connectivity, and optionally
        nodal displacement fields.
    shader : object
        Backend-owned material or shader definition for the mesh.
    displacement_keys : Sequence[str] or None, optional
        Names of the three displacement components. ``None`` omits motion.
    extract_surface : bool, optional
        Extract the exterior surface before conversion. Select this for a
        volume simulation.

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
    A :class:`Mesh3D` has one topology. Split simulations with several
    connectivity tables into separate render meshes before conversion.
    """
    prepared = enforce_mesh_convention(sim_data)
    if extract_surface:
        prepared = extract_surf_mesh(prepared)
    if prepared.coords is None or prepared.connect is None:
        raise ValueError("SimData must provide coordinates and connectivity.")
    if len(prepared.connect) != 1:
        raise ValueError("SimData must have exactly one connectivity table.")
    connectivity = next(iter(prepared.connect.values()))
    element_type = _element_type_from_nodes(connectivity.shape[1])
    displacements = _displacements_from_simdata(prepared, displacement_keys)
    return Mesh3D(
        element_type=element_type,
        coords=np.ascontiguousarray(prepared.coords[:, :3], dtype=np.float64),
        connectivity=np.ascontiguousarray(connectivity, dtype=np.uintp),
        shader=shader,
        displacements=displacements,
    )


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
            f"No render element type for {nodes_per_element} nodes per element.",
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
        fields = [np.asarray(sim_data.node_vars[key], dtype=np.float64)
                  for key in displacement_keys]
    except KeyError as error:
        raise ValueError(f"Missing displacement field {error.args[0]!r}.") from error
    if any(field.ndim != 2 for field in fields):
        raise ValueError("Displacement fields must have shape (nodes, frames).")
    displacements = np.stack(fields, axis=2).transpose(1, 0, 2)
    if displacements.shape[2] == 2:
        displacements = np.pad(displacements, ((0, 0), (0, 0), (0, 1)))
    return np.ascontiguousarray(displacements)


__all__ = ["mesh_from_simdata"]
