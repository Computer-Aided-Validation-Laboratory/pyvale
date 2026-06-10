# ==============================================================================
# pyvale: the python validation engine
# License: MIT
# Copyright (C) 2025 The Computer Aided Validation Team
# ==============================================================================

"""Shared mesh convention and surface-extraction tools for ``SimData``."""

from __future__ import annotations

from dataclasses import dataclass
import numpy as np

from pyvale.dataio.simdata import SimData

_SURFACE_NODE_COUNTS = frozenset((3, 4, 6, 7, 8, 9))
_VOLUME_NODE_COUNTS = frozenset((4, 8, 10, 20, 27))
_SUPPORTED_NODE_COUNTS = _SURFACE_NODE_COUNTS | _VOLUME_NODE_COUNTS
_TOL = 1.0e-12


@dataclass(slots=True)
class MeshConventionCheck:
    """Structured report describing mesh convention compliance."""

    is_zero_based: bool
    has_ccw_winding: bool
    is_right_handed: bool
    has_valid_connectivity: bool
    has_row_major_connectivity: bool
    failed_checks: tuple[str, ...]
    connectivity_failures: dict[str, tuple[str, ...]]

    @property
    def is_valid(self) -> bool:
        return not self.failed_checks


def check_mesh_convention(mesh_in: SimData) -> MeshConventionCheck:
    """Checks a mesh against the shared pyvale mesh convention."""

    if mesh_in.connect is None:
        return MeshConventionCheck(True, True, True, True, True, tuple(), {})
    if mesh_in.coords is None:
        raise ValueError("Mesh convention checks require 'coords' to be set.")

    zero_based = True
    ccw = True
    right_handed = True
    valid_connectivity = True
    row_major = True
    per_table: dict[str, tuple[str, ...]] = {}
    shift_all = _infer_mesh_zero_based_shift(mesh_in, mesh_in.coords.shape[0])

    for name, connect_raw in mesh_in.connect.items():
        connect = _coerce_connect_array(connect_raw, name)
        failures: list[str] = []

        if _should_transpose_connectivity(connect, name, mesh_in):
            row_major = False
            failures.append("row_major_connectivity")
            connect = connect.T

        legacy_connect = _table_needs_zero_based_shift(
            connect,
            mesh_in.coords.shape[0],
            shift_all,
        )
        if legacy_connect:
            zero_based = False
            failures.append("zero_based_indexing")

        connect_check = connect
        if legacy_connect:
            connect_check = connect - 1

        if not _check_indices_zero_based(connect_check, mesh_in.coords.shape[0]):
            valid_connectivity = False
            failures.append("connectivity_indices")
        else:
            connect_check = _normalise_legacy_connectivity_order(connect_check, legacy_connect)

            if not _check_ccw_winding_table(connect_check, mesh_in.coords):
                ccw = False
                failures.append("ccw_winding")

            if not _check_right_handed_table(connect_check, mesh_in.coords):
                right_handed = False
                failures.append("right_handed_geometry")

        per_table[name] = tuple(failures)

    failed_checks: list[str] = []
    if not zero_based:
        failed_checks.append("zero_based_indexing")
    if not ccw:
        failed_checks.append("ccw_winding")
    if not right_handed:
        failed_checks.append("right_handed_geometry")
    if not valid_connectivity:
        failed_checks.append("connectivity_indices")
    if not row_major:
        failed_checks.append("row_major_connectivity")

    return MeshConventionCheck(
        is_zero_based=zero_based,
        has_ccw_winding=ccw,
        is_right_handed=right_handed,
        has_valid_connectivity=valid_connectivity,
        has_row_major_connectivity=row_major,
        failed_checks=tuple(failed_checks),
        connectivity_failures=per_table,
    )


def enforce_mesh_convention(mesh_in: SimData) -> SimData:
    """Normalises a mesh to the pyvale mesh convention."""

    report = check_mesh_convention(mesh_in)
    if report.is_valid:
        return mesh_in

    if mesh_in.connect is None:
        return mesh_in
    if mesh_in.coords is None:
        raise ValueError("Mesh convention enforcement requires 'coords' to be set.")

    connect_out: dict[str, np.ndarray] = {}
    shift_all = _infer_mesh_zero_based_shift(mesh_in, mesh_in.coords.shape[0])

    for name, connect_raw in mesh_in.connect.items():
        connect = _coerce_connect_array(connect_raw, name)

        if _should_transpose_connectivity(connect, name, mesh_in):
            connect = connect.T

        legacy_connect = _table_needs_zero_based_shift(
            connect,
            mesh_in.coords.shape[0],
            shift_all,
        )
        if legacy_connect:
            connect = connect - 1

        if not _check_indices_zero_based(connect, mesh_in.coords.shape[0]):
            raise ValueError(
                "Connectivity table "
                f"'{name}' contains indices outside the coordinate array after "
                "0-based normalization."
            )

        connect = _normalise_legacy_connectivity_order(connect, legacy_connect)
        connect = _enforce_ccw_winding_table(connect, mesh_in.coords)
        connect = _enforce_right_handed_table(connect, mesh_in.coords)

        if not _check_indices_zero_based(connect, mesh_in.coords.shape[0]):
            raise ValueError(
                "Connectivity table "
                f"'{name}' became invalid during mesh convention enforcement."
            )

        connect_out[name] = np.ascontiguousarray(connect, dtype=np.int64)

    return _copy_sim_data(mesh_in, connect=connect_out)


def check_cw_winding(mesh_in: SimData) -> bool:
    """Checks whether all supported surface/2D elements are wound clockwise."""

    if mesh_in.connect is None:
        return True
    if mesh_in.coords is None:
        raise ValueError("Clockwise winding checks require 'coords' to be set.")

    shift_all = _infer_mesh_zero_based_shift(mesh_in, mesh_in.coords.shape[0])
    for connect in mesh_in.connect.values():
        connect_row_major = _as_row_major_zero_based(
            connect,
            mesh_in.coords.shape[0],
            shift_all=shift_all,
        )
        if not _check_cw_winding_table(connect_row_major, mesh_in.coords):
            return False
    return True


def check_ccw_winding(mesh_in: SimData) -> bool:
    """Checks whether all supported surface/2D elements are wound counter-clockwise."""

    if mesh_in.connect is None:
        return True
    if mesh_in.coords is None:
        raise ValueError("CCW winding checks require 'coords' to be set.")

    shift_all = _infer_mesh_zero_based_shift(mesh_in, mesh_in.coords.shape[0])
    for connect in mesh_in.connect.values():
        connect_row_major = _as_row_major_zero_based(
            connect,
            mesh_in.coords.shape[0],
            shift_all=shift_all,
        )
        if not _check_ccw_winding_table(connect_row_major, mesh_in.coords):
            return False
    return True


def enforce_cw_winding(mesh_in: SimData) -> SimData:
    """Returns a copy of ``mesh_in`` with supported 2D/surface elements wound CW."""

    if mesh_in.connect is None:
        return mesh_in
    if mesh_in.coords is None:
        raise ValueError("Clockwise winding enforcement requires 'coords' to be set.")

    connect_out: dict[str, np.ndarray] = {}
    changed = False
    shift_all = _infer_mesh_zero_based_shift(mesh_in, mesh_in.coords.shape[0])

    for name, connect in mesh_in.connect.items():
        connect_row_major = _as_row_major_zero_based(
            connect,
            mesh_in.coords.shape[0],
            shift_all=shift_all,
        )
        connect_out[name] = _enforce_cw_winding_table(connect_row_major, mesh_in.coords)
        changed = changed or not np.array_equal(connect_out[name], connect_row_major)

    if not changed:
        return mesh_in

    return _copy_sim_data(mesh_in, connect=connect_out)


def enforce_ccw_winding(mesh_in: SimData) -> SimData:
    """Returns a copy of ``mesh_in`` with supported 2D/surface elements wound CCW."""

    if mesh_in.connect is None:
        return mesh_in
    if mesh_in.coords is None:
        raise ValueError("CCW winding enforcement requires 'coords' to be set.")

    connect_out: dict[str, np.ndarray] = {}
    changed = False
    shift_all = _infer_mesh_zero_based_shift(mesh_in, mesh_in.coords.shape[0])

    for name, connect in mesh_in.connect.items():
        connect_row_major = _as_row_major_zero_based(
            connect,
            mesh_in.coords.shape[0],
            shift_all=shift_all,
        )
        connect_out[name] = _enforce_ccw_winding_table(connect_row_major, mesh_in.coords)
        changed = changed or not np.array_equal(connect_out[name], connect_row_major)

    if not changed:
        return mesh_in

    return _copy_sim_data(mesh_in, connect=connect_out)


def extract_surf_mesh(
    mesh_in: SimData,
    enforce_convention: bool = True,
) -> SimData:
    """Extracts the external surface mesh from supported 3D volume elements."""

    if mesh_in.connect is None:
        raise ValueError("Surface extraction requires connectivity tables.")
    if mesh_in.coords is None:
        raise ValueError("Surface extraction requires 'coords' to be set.")

    connect_norm: dict[str, np.ndarray] = {}
    source_zero_based = True
    source_row_major = True
    shift_all = _infer_mesh_zero_based_shift(mesh_in, mesh_in.coords.shape[0])

    for name, connect_raw in mesh_in.connect.items():
        connect = _coerce_connect_array(connect_raw, name)
        if _should_transpose_connectivity(connect, name, mesh_in):
            source_row_major = False
            connect = connect.T
        legacy_connect = _table_needs_zero_based_shift(
            connect,
            mesh_in.coords.shape[0],
            shift_all,
        )
        if legacy_connect:
            source_zero_based = False
            connect = connect - 1
        if not _check_indices_zero_based(connect, mesh_in.coords.shape[0]):
            raise ValueError(
                f"Connectivity table '{name}' contains invalid indices for surface extraction."
            )
        connect = _normalise_legacy_connectivity_order(connect, legacy_connect)
        connect_norm[name] = _enforce_right_handed_table(
            _enforce_ccw_winding_table(connect, mesh_in.coords),
            mesh_in.coords,
        )

    surf_connect_global: dict[str, np.ndarray] = {}
    surf_elem_sources: dict[str, np.ndarray] = {}
    surf_node_inds = np.array([], dtype=np.int64)

    for name, connect in connect_norm.items():
        surf_faces = _extract_surface_faces_from_table(connect)
        surf_connect_global[name] = surf_faces
        surf_elem_sources[name] = _extract_parent_elem_inds(connect)
        if surf_faces.size:
            surf_node_inds = np.union1d(surf_node_inds, np.unique(surf_faces))

    surf_coords = np.ascontiguousarray(mesh_in.coords[surf_node_inds], dtype=mesh_in.coords.dtype)
    coord_remap = np.full(mesh_in.coords.shape[0], -1, dtype=np.int64)
    coord_remap[surf_node_inds] = np.arange(surf_node_inds.shape[0], dtype=np.int64)

    surf_connect_local: dict[str, np.ndarray] = {}
    for name, surf_faces in surf_connect_global.items():
        surf_connect_local[name] = coord_remap[surf_faces]

    surf_mesh = _copy_sim_data(mesh_in)
    surf_mesh.coords = surf_coords
    surf_mesh.connect = surf_connect_local

    if mesh_in.node_vars is not None:
        surf_mesh.node_vars = {
            name: values[surf_node_inds, :]
            for name, values in mesh_in.node_vars.items()
        }

    if mesh_in.elem_vars is not None:
        surf_mesh.elem_vars = {}
        for (name, block_id), values in mesh_in.elem_vars.items():
            connect_key = f"connect{block_id}"
            if connect_key in surf_elem_sources:
                surf_mesh.elem_vars[(name, block_id)] = values[surf_elem_sources[connect_key], :]

    if not enforce_convention:
        surf_mesh = _restore_source_connectivity_style(
            surf_mesh,
            one_based=not source_zero_based,
            transposed=not source_row_major,
        )
        return surf_mesh

    return enforce_mesh_convention(surf_mesh)


def _copy_sim_data(mesh_in: SimData, connect: dict[str, np.ndarray] | None = None) -> SimData:
    mesh_out = SimData(
        num_spat_dims=mesh_in.num_spat_dims,
        time=mesh_in.time,
        coords=mesh_in.coords,
        connect=mesh_in.connect if connect is None else connect,
        side_sets=mesh_in.side_sets,
        node_vars=mesh_in.node_vars,
        elem_vars=mesh_in.elem_vars,
        glob_vars=mesh_in.glob_vars,
    )
    return mesh_out


def _coerce_connect_array(connect: np.ndarray, name: str) -> np.ndarray:
    array = np.asarray(connect)
    if array.ndim != 2:
        raise ValueError(
            f"Connectivity table '{name}' must be 2D, got shape {array.shape}."
        )
    return np.ascontiguousarray(array, dtype=np.int64)


def _supported_nodes_per_elem(nodes_per_elem: int) -> None:
    if nodes_per_elem not in _SUPPORTED_NODE_COUNTS:
        raise NotImplementedError(
            "Mesh convention tools do not support elements with "
            f"{nodes_per_elem} nodes."
        )


def _should_transpose_connectivity(
    connect: np.ndarray,
    connect_name: str | None = None,
    mesh_in: SimData | None = None,
) -> bool:
    rows_supported = connect.shape[0] in _SUPPORTED_NODE_COUNTS
    cols_supported = connect.shape[1] in _SUPPORTED_NODE_COUNTS

    if rows_supported and not cols_supported:
        return True
    if cols_supported and not rows_supported:
        return False
    if cols_supported and rows_supported:
        block_rows = _get_elem_var_block_rows(connect_name, mesh_in)
        if block_rows is not None:
            row_match = connect.shape[0] in block_rows
            col_match = connect.shape[1] in block_rows
            if row_match and not col_match:
                return False
            if col_match and not row_match:
                return True
        if mesh_in is not None and mesh_in.coords is not None:
            return _score_orientation(connect.T, mesh_in.coords) > _score_orientation(
                connect,
                mesh_in.coords,
            )
        return False
    raise NotImplementedError(
        "Could not infer connectivity orientation from shape "
        f"{connect.shape}. Expected a supported nodes-per-element dimension."
    )


def _needs_zero_based_shift(connect: np.ndarray, num_coords: int) -> bool:
    if connect.size == 0:
        return False
    if np.any(connect < 0):
        return False
    if np.any(connect == 0):
        return False
    return bool(np.any(connect >= num_coords))


def _is_ambiguously_positive(connect: np.ndarray, num_coords: int) -> bool:
    if connect.size == 0:
        return False
    return bool(
        np.all(connect >= 1)
        and np.all(connect < num_coords)
        and not np.any(connect == 0)
    )


def _infer_mesh_zero_based_shift(mesh_in: SimData, num_coords: int) -> bool:
    any_zero_based = False
    any_definitely_one_based = False

    for connect_name, connect_raw in mesh_in.connect.items():
        connect = np.asarray(connect_raw, dtype=np.int64)
        if _should_transpose_connectivity(connect, connect_name, mesh_in):
            connect = connect.T
        if np.any(connect == 0):
            any_zero_based = True
        if _needs_zero_based_shift(connect, num_coords):
            any_definitely_one_based = True

    if any_zero_based and any_definitely_one_based:
        raise ValueError(
            "Mixed zero-based and one-based connectivity tables detected in the same mesh."
        )

    return any_definitely_one_based and not any_zero_based


def _table_needs_zero_based_shift(
    connect: np.ndarray,
    num_coords: int,
    shift_all: bool,
) -> bool:
    if _needs_zero_based_shift(connect, num_coords):
        return True
    return shift_all and _is_ambiguously_positive(connect, num_coords)


def _get_elem_var_block_rows(
    connect_name: str | None,
    mesh_in: SimData | None,
) -> set[int] | None:
    if connect_name is None or mesh_in is None or mesh_in.elem_vars is None:
        return None

    try:
        block_id = int(connect_name.replace("connect", ""))
    except ValueError:
        return None

    row_counts = {
        values.shape[0]
        for (_field_name, field_block), values in mesh_in.elem_vars.items()
        if field_block == block_id
    }
    return row_counts or None


def _check_indices_zero_based(connect: np.ndarray, num_coords: int) -> bool:
    if connect.size == 0:
        return True
    valid_mask = np.logical_and(connect >= 0, connect < num_coords)
    return bool(np.all(valid_mask))


def _as_row_major_zero_based(
    connect: np.ndarray,
    num_coords: int,
    *,
    shift_all: bool = False,
) -> np.ndarray:
    connect_out = np.asarray(connect, dtype=np.int64)
    if connect_out.ndim != 2:
        raise ValueError(f"Connectivity table must be 2D, got shape {connect_out.shape}.")
    if _should_transpose_connectivity(connect_out):
        connect_out = connect_out.T
    legacy_connect = _table_needs_zero_based_shift(connect_out, num_coords, shift_all)
    if legacy_connect:
        connect_out = connect_out - 1
    connect_out = _normalise_legacy_connectivity_order(connect_out, legacy_connect)
    return np.ascontiguousarray(connect_out, dtype=np.int64)


def _score_orientation(connect_row_major: np.ndarray, coords: np.ndarray) -> int:
    num_coords = coords.shape[0]
    connect_eval = np.asarray(connect_row_major, dtype=np.int64)

    if connect_eval.ndim != 2 or connect_eval.shape[1] not in _SUPPORTED_NODE_COUNTS:
        return -1

    if _needs_zero_based_shift(connect_eval, num_coords):
        connect_eval = connect_eval - 1

    if not _check_indices_zero_based(connect_eval, num_coords):
        return -1

    score = 0
    for row in connect_eval:
        if np.unique(row).shape[0] != row.shape[0]:
            continue

        try:
            metric = _handedness_metric(row, coords)
        except ValueError:
            continue

        if metric is not None and abs(metric) > _TOL:
            score += 1

    return score


def _normalise_legacy_connectivity_order(
    connect: np.ndarray,
    legacy_connect: bool,
) -> np.ndarray:
    if not legacy_connect:
        return connect

    nodes_per_elem = connect.shape[1]
    if nodes_per_elem == 20:
        perm = np.array((0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 16, 17, 18, 19, 12, 13, 14, 15))
        return np.ascontiguousarray(connect[:, perm], dtype=np.int64)
    if nodes_per_elem == 27:
        perm = np.array((0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 16, 17, 18, 19, 12, 13, 14, 15, 26, 24, 25, 20, 21, 22, 23))
        return np.ascontiguousarray(connect[:, perm], dtype=np.int64)
    return connect


def _get_corner_indices(nodes_per_elem: int) -> np.ndarray:
    _supported_nodes_per_elem(nodes_per_elem)
    if nodes_per_elem in (3, 6, 7):
        return np.array((0, 1, 2), dtype=np.int64)
    if nodes_per_elem in (4, 8, 9):
        return np.array((0, 1, 2, 3), dtype=np.int64)
    if nodes_per_elem in (10,):
        return np.array((0, 1, 2, 3), dtype=np.int64)
    if nodes_per_elem in (20, 27):
        return np.array((0, 1, 2, 3, 4, 5, 6, 7), dtype=np.int64)
    return np.arange(nodes_per_elem, dtype=np.int64)


def _active_coord_axes(coords: np.ndarray) -> np.ndarray:
    axis_range = np.ptp(coords, axis=0)
    active = np.flatnonzero(axis_range > _TOL)
    if active.shape[0] < 2:
        raise ValueError("At least two active coordinate axes are required.")
    return active[:2]


def _polygon_signed_area(coords_elem: np.ndarray) -> float:
    axes = _active_coord_axes(coords_elem)
    xy = coords_elem[:, axes]
    rolled = np.roll(xy, -1, axis=0)
    return 0.5 * np.sum(xy[:, 0] * rolled[:, 1] - rolled[:, 0] * xy[:, 1])


def _is_coplanar(coords_elem: np.ndarray) -> bool:
    centred = coords_elem - np.mean(coords_elem, axis=0)
    return np.linalg.matrix_rank(centred, tol=_TOL) <= 2


def _tet_signed_volume(coords_elem: np.ndarray) -> float:
    return float(
        np.linalg.det(
            np.column_stack((
                coords_elem[1] - coords_elem[0],
                coords_elem[2] - coords_elem[0],
                coords_elem[3] - coords_elem[0],
            ))
        )
    )


def _hex_signed_volume(coords_elem: np.ndarray) -> float:
    return float(
        np.linalg.det(
            np.column_stack((
                coords_elem[1] - coords_elem[0],
                coords_elem[3] - coords_elem[0],
                coords_elem[4] - coords_elem[0],
            ))
        )
    )


def _winding_metric(connect_row: np.ndarray, coords: np.ndarray) -> float | None:
    nodes_per_elem = connect_row.shape[0]
    if nodes_per_elem not in _SURFACE_NODE_COUNTS:
        return None
    corner_inds = _get_corner_indices(nodes_per_elem)
    coords_elem = coords[connect_row[corner_inds]]
    if not _is_coplanar(coords_elem):
        return None
    return _polygon_signed_area(coords_elem)


def _handedness_metric(connect_row: np.ndarray, coords: np.ndarray) -> float | None:
    nodes_per_elem = connect_row.shape[0]
    if nodes_per_elem in _SURFACE_NODE_COUNTS:
        metric = _winding_metric(connect_row, coords)
        if metric is not None:
            return metric
    corner_inds = _get_corner_indices(nodes_per_elem)
    coords_elem = coords[connect_row[corner_inds]]
    if nodes_per_elem in (4, 10):
        return _tet_signed_volume(coords_elem)
    if nodes_per_elem in (8, 20, 27):
        return _hex_signed_volume(coords_elem)
    raise NotImplementedError(
        f"Handedness checks are not implemented for {nodes_per_elem}-node elements."
    )


def _check_ccw_winding_table(connect: np.ndarray, coords: np.ndarray) -> bool:
    for row in connect:
        metric = _winding_metric(row, coords)
        if metric is None:
            continue
        if metric <= 0.0:
            return False
    return True


def _check_cw_winding_table(connect: np.ndarray, coords: np.ndarray) -> bool:
    for row in connect:
        metric = _winding_metric(row, coords)
        if metric is None:
            continue
        if metric >= 0.0:
            return False
    return True


def _check_right_handed_table(connect: np.ndarray, coords: np.ndarray) -> bool:
    for row in connect:
        metric = _handedness_metric(row, coords)
        if metric is None:
            continue
        if metric <= 0.0:
            return False
    return True


def _reverse_surface_row(connect_row: np.ndarray) -> np.ndarray:
    nodes_per_elem = connect_row.shape[0]
    perms = {
        3: np.array((0, 2, 1)),
        4: np.array((0, 3, 2, 1)),
        6: np.array((0, 2, 1, 5, 4, 3)),
        7: np.array((0, 2, 1, 5, 4, 3, 6)),
        8: np.array((0, 3, 2, 1, 7, 6, 5, 4)),
        9: np.array((0, 3, 2, 1, 7, 6, 5, 4, 8)),
    }
    _supported_nodes_per_elem(nodes_per_elem)
    if nodes_per_elem not in perms:
        raise NotImplementedError(
            f"CCW/CW winding enforcement is not implemented for {nodes_per_elem}-node elements."
        )
    return connect_row[perms[nodes_per_elem]]


def _reverse_handedness_row(connect_row: np.ndarray) -> np.ndarray:
    nodes_per_elem = connect_row.shape[0]
    perms = {
        4: np.array((0, 2, 1, 3)),
        10: np.array((0, 2, 1, 3, 6, 5, 4, 7, 9, 8)),
        8: np.array((0, 3, 2, 1, 4, 7, 6, 5)),
        20: np.array((0, 3, 2, 1, 4, 7, 6, 5, 11, 10, 9, 8, 15, 14, 13, 12, 16, 19, 18, 17)),
        27: np.array((0, 3, 2, 1, 4, 7, 6, 5, 11, 10, 9, 8, 15, 14, 13, 12, 16, 19, 18, 17, 20, 21, 22, 23, 24, 25, 26)),
    }
    if nodes_per_elem not in perms:
        raise NotImplementedError(
            f"Right-handed enforcement is not implemented for {nodes_per_elem}-node elements."
        )
    return connect_row[perms[nodes_per_elem]]


def _enforce_ccw_winding_table(connect: np.ndarray, coords: np.ndarray) -> np.ndarray:
    connect_out = np.copy(connect)
    for idx, row in enumerate(connect_out):
        metric = _winding_metric(row, coords)
        if metric is not None and metric < 0.0:
            connect_out[idx, :] = _reverse_surface_row(row)
    return np.ascontiguousarray(connect_out, dtype=np.int64)


def _enforce_cw_winding_table(connect: np.ndarray, coords: np.ndarray) -> np.ndarray:
    connect_out = np.copy(connect)
    for idx, row in enumerate(connect_out):
        metric = _winding_metric(row, coords)
        if metric is not None and metric > 0.0:
            connect_out[idx, :] = _reverse_surface_row(row)
    return np.ascontiguousarray(connect_out, dtype=np.int64)


def _enforce_right_handed_table(connect: np.ndarray, coords: np.ndarray) -> np.ndarray:
    connect_out = np.copy(connect)
    for idx, row in enumerate(connect_out):
        metric = _handedness_metric(row, coords)
        if metric is not None and metric < 0.0:
            if row.shape[0] in _SURFACE_NODE_COUNTS and _is_coplanar(coords[row[_get_corner_indices(row.shape[0])]]):
                connect_out[idx, :] = _reverse_surface_row(row)
            else:
                connect_out[idx, :] = _reverse_handedness_row(row)
    return np.ascontiguousarray(connect_out, dtype=np.int64)


def _get_surface_map(nodes_per_elem: int) -> np.ndarray:
    if nodes_per_elem == 4:  # TET4
        return np.array(((0, 1, 2),
                         (0, 3, 1),
                         (0, 2, 3),
                         (1, 3, 2)), dtype=np.int64)
    if nodes_per_elem == 8:  # HEX8
        return np.array(((0, 1, 2, 3),
                         (0, 3, 7, 4),
                         (4, 7, 6, 5),
                         (1, 5, 6, 2),
                         (0, 4, 5, 1),
                         (2, 6, 7, 3)), dtype=np.int64)
    if nodes_per_elem == 10:  # TET10
        return np.array(((0, 1, 2, 4, 5, 6),
                         (0, 3, 1, 7, 8, 4),
                         (0, 2, 3, 6, 9, 7),
                         (1, 3, 2, 8, 9, 5)), dtype=np.int64)
    if nodes_per_elem == 20:  # HEX20
        return np.array(((0, 1, 2, 3, 8, 9, 10, 11),
                         (0, 3, 7, 4, 11, 15, 19, 16),
                         (4, 7, 6, 5, 15, 14, 13, 12),
                         (1, 5, 6, 2, 17, 13, 18, 9),
                         (0, 4, 5, 1, 16, 12, 17, 8),
                         (2, 6, 7, 3, 18, 14, 19, 10)), dtype=np.int64)
    if nodes_per_elem == 27:  # HEX27
        return np.array(((0, 1, 2, 3, 8, 9, 10, 11, 21),
                         (0, 3, 7, 4, 11, 15, 19, 16, 23),
                         (4, 7, 6, 5, 15, 14, 13, 12, 22),
                         (1, 5, 6, 2, 17, 13, 18, 9, 24),
                         (0, 4, 5, 1, 16, 12, 17, 8, 25),
                         (2, 6, 7, 3, 18, 14, 19, 10, 26)), dtype=np.int64)
    raise NotImplementedError(
        "Surface extraction is only implemented for tet and hex element families."
    )


def _extract_surface_faces_from_table(connect: np.ndarray) -> np.ndarray:
    nodes_per_elem = connect.shape[1]
    face_map = _get_surface_map(nodes_per_elem)
    faces_wound = connect[:, face_map]
    faces_flat_wound = faces_wound.reshape((-1, face_map.shape[1]))
    faces_flat_sorted = np.sort(faces_flat_wound, axis=1)

    (_, unique_inds, unique_counts) = np.unique(
        faces_flat_sorted,
        axis=0,
        return_index=True,
        return_counts=True,
    )
    ext_face_inds = unique_inds[unique_counts == 1]
    return np.ascontiguousarray(faces_flat_wound[ext_face_inds], dtype=np.int64)


def _extract_parent_elem_inds(connect: np.ndarray) -> np.ndarray:
    nodes_per_elem = connect.shape[1]
    face_map = _get_surface_map(nodes_per_elem)
    faces_per_elem = face_map.shape[0]
    faces_wound = connect[:, face_map]
    faces_flat_wound = faces_wound.reshape((-1, face_map.shape[1]))
    faces_flat_sorted = np.sort(faces_flat_wound, axis=1)

    (_, unique_inds, unique_counts) = np.unique(
        faces_flat_sorted,
        axis=0,
        return_index=True,
        return_counts=True,
    )
    ext_face_inds = unique_inds[unique_counts == 1]
    return np.ascontiguousarray(ext_face_inds // faces_per_elem, dtype=np.int64)


def _restore_source_connectivity_style(
    mesh_in: SimData,
    *,
    one_based: bool,
    transposed: bool,
) -> SimData:
    if mesh_in.connect is None:
        return mesh_in

    connect_out: dict[str, np.ndarray] = {}
    for name, connect in mesh_in.connect.items():
        connect_fmt = np.copy(connect)
        if one_based:
            connect_fmt = connect_fmt + 1
        if transposed:
            connect_fmt = connect_fmt.T
        connect_out[name] = np.ascontiguousarray(connect_fmt, dtype=np.int64)

    return _copy_sim_data(mesh_in, connect=connect_out)
