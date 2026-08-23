# ==============================================================================
# pyvale: the python validation engine
# License: MIT
# Copyright (C) 2025 The Computer Aided Validation Team
# ==============================================================================

"""PyVale ``SimData`` adapter for Riley's mesh-convention tools."""

from __future__ import annotations

import numpy as np

from riley.python import meshconv as _core

from pyvale.dataio.simdata import EMeshType as PyValeMeshType
from pyvale.dataio.simdata import SimData


CheckCode = _core.CheckCode
MeshConventionCheck = _core.MeshConventionCheck
EElementType = _core.EElementType
ElementSpec = _core.ElementSpec
ELEMENT_SPECS = _core.ELEMENT_SPECS
ELEMENT_SYMMETRIES = _core.ELEMENT_SYMMETRIES
MeshConvention = _core.MeshConvention
MeshConventionInferenceError = _core.MeshConventionInferenceError

# PyVale's historical Exodus fixtures use this documented source layout.  It
# is adapter-owned metadata, not a Riley convention special case.
PYVALE_EXODUS_MESH_CONVENTION = MeshConvention({
    _core.EElementType.HEX20: (
        0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11,
        16, 17, 18, 19, 12, 13, 14, 15,
    ),
    _core.EElementType.HEX27: (
        0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11,
        16, 17, 18, 19, 12, 13, 14, 15,
        25, 24, 26, 23, 21, 22, 20,
    ),
})


def check_mesh_convention(
    mesh_in: SimData,
    source_convention: MeshConvention | None = None,
) -> MeshConventionCheck:
    return _core.check_mesh_convention(
        _to_riley_mesh(mesh_in),
        _resolve_source_convention(mesh_in, source_convention),
    )


def enforce_mesh_convention(
    mesh_in: SimData,
    source_convention: MeshConvention | None = None,
) -> SimData:
    mesh_out = _core.enforce_mesh_convention(
        _to_riley_mesh(mesh_in),
        _resolve_source_convention(mesh_in, source_convention),
    )
    return _from_riley_mesh(mesh_out, mesh_in.num_spat_dims)


def check_cw_winding(mesh_in: SimData) -> bool:
    return _core.check_cw_winding(_to_riley_mesh(mesh_in))


def check_ccw_winding(mesh_in: SimData) -> bool:
    return _core.check_ccw_winding(_to_riley_mesh(mesh_in))


def enforce_cw_winding(mesh_in: SimData) -> SimData:
    mesh_out = _core.enforce_cw_winding(_to_riley_mesh(mesh_in))
    return _from_riley_mesh(mesh_out, mesh_in.num_spat_dims)


def enforce_ccw_winding(mesh_in: SimData) -> SimData:
    mesh_out = _core.enforce_ccw_winding(_to_riley_mesh(mesh_in))
    return _from_riley_mesh(mesh_out, mesh_in.num_spat_dims)


def is_mesh_2d(mesh_in: SimData) -> bool:
    return _core.is_mesh_2d(_to_riley_mesh(mesh_in))


def is_volume_mesh(mesh_in: SimData) -> bool:
    return _core.is_volume_mesh(_to_riley_mesh(mesh_in))


def extract_surf_mesh(
    mesh_in: SimData,
    enforce_convention: bool = True,
) -> SimData:
    mesh_out = _core.extract_surf_mesh(
        _to_riley_mesh(mesh_in),
        enforce_convention=enforce_convention,
    )
    return _from_riley_mesh(mesh_out, mesh_in.num_spat_dims)


def extract_surf_between(
    mesh_in: SimData,
    point: np.ndarray | list[float] | tuple[float, ...],
    normal: np.ndarray | list[float] | tuple[float, ...],
    distance: float | None = None,
    tolerance: float = 1.0e-6,
    enforce_convention: bool = True,
) -> SimData:
    mesh_out = _core.extract_surf_between(
        _to_riley_mesh(mesh_in),
        point,
        normal,
        distance=distance,
        tolerance=tolerance,
        enforce_convention=enforce_convention,
    )
    return _from_riley_mesh(mesh_out, mesh_in.num_spat_dims)


def _to_riley_mesh(mesh_in: SimData) -> _core.SimData:
    return _core.SimData(
        coords=mesh_in.coords,
        connect=mesh_in.connect,
        mesh_type=_to_riley_mesh_type(mesh_in.mesh_type),
        time=mesh_in.time,
        side_sets=mesh_in.side_sets,
        node_vars=mesh_in.node_vars,
        elem_vars=mesh_in.elem_vars,
        glob_vars=mesh_in.glob_vars,
    )


def _resolve_source_convention(
    mesh_in: SimData,
    source_convention: MeshConvention | None,
) -> MeshConvention | None:
    if source_convention is not None:
        return source_convention
    if mesh_in.coords is None or mesh_in.connect is None:
        return None
    for connect in mesh_in.connect.values():
        connect_array = np.asarray(connect, dtype=np.int64)
        if (
            _core._should_transpose_connectivity(connect_array)
            or int(connect_array.min()) == 1
        ):
            return PYVALE_EXODUS_MESH_CONVENTION
    return None


def _from_riley_mesh(mesh_in: _core.SimData, num_spat_dims: int) -> SimData:
    return SimData(
        num_spat_dims=num_spat_dims,
        mesh_type=_to_pyvale_mesh_type(mesh_in.mesh_type),
        time=mesh_in.time,
        coords=mesh_in.coords,
        connect=mesh_in.connect,
        side_sets=mesh_in.side_sets,
        node_vars=mesh_in.node_vars,
        elem_vars=mesh_in.elem_vars,
        glob_vars=mesh_in.glob_vars,
    )


def _to_riley_mesh_type(
    mesh_type: PyValeMeshType | None,
) -> _core.EMeshType | None:
    if mesh_type is PyValeMeshType.VOL:
        return _core.EMeshType.VOL
    if mesh_type is PyValeMeshType.SURF:
        return _core.EMeshType.SURF
    return None


def _to_pyvale_mesh_type(
    mesh_type: _core.EMeshType | None,
) -> PyValeMeshType | None:
    if mesh_type is _core.EMeshType.VOL:
        return PyValeMeshType.VOL
    if mesh_type is _core.EMeshType.SURF:
        return PyValeMeshType.SURF
    return None


__all__ = [
    "CheckCode",
    "MeshConventionCheck",
    "check_mesh_convention",
    "enforce_mesh_convention",
    "check_cw_winding",
    "check_ccw_winding",
    "enforce_cw_winding",
    "enforce_ccw_winding",
    "is_mesh_2d",
    "is_volume_mesh",
    "extract_surf_mesh",
    "extract_surf_between",
]
