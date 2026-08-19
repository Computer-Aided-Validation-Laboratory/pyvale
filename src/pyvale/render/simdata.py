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

from .mesh import EElementType, Mesh


def mesh_from_simdata(
    sim_data: SimData,
    shader: object,
    displacement_keys: Sequence[str] | None = None,
    extract_surface: bool = False,
) -> Mesh:
    """Build one ``render.Mesh`` from a convention-normalised ``SimData``.

    ``extract_surface`` should be selected for a volume simulation.  A source
    with more than one connectivity table is rejected because a ``Mesh`` has
    one element topology; callers can split it into multiple render meshes.
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
    return Mesh(
        element_type=element_type,
        coords=np.ascontiguousarray(prepared.coords[:, :3], dtype=np.float64),
        connectivity=np.ascontiguousarray(connectivity, dtype=np.uintp),
        shader=shader,
        displacements=displacements,
    )


def _element_type_from_nodes(nodes_per_element: int) -> EElementType:
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
    if displacement_keys is None:
        return None
    if sim_data.node_vars is None or len(displacement_keys) != 3:
        raise ValueError("Three displacement keys are required.")
    try:
        fields = [np.asarray(sim_data.node_vars[key], dtype=np.float64)
                  for key in displacement_keys]
    except KeyError as error:
        raise ValueError(f"Missing displacement field {error.args[0]!r}.") from error
    if any(field.ndim != 2 for field in fields):
        raise ValueError("Displacement fields must have shape (nodes, frames).")
    return np.ascontiguousarray(np.stack(fields, axis=2).transpose(1, 0, 2))


__all__ = ["mesh_from_simdata"]
