# --------------------------------------------------------------------------
# Riley: A High Performance Rasteriser for DIC UQ
#
# Copyright (c) 2025-2026 scepticalrabbit (Lloyd Fletcher)
# Licensed under the MIT License (see LICENSE file for details)
#
# Authors: scepticalrabbit (Lloyd Fletcher)
# --------------------------------------------------------------------------
"""Implementation of Riley's mesh convention tools.

The public interface lives in :mod:`riley.python.meshconv`.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from enum import Enum, StrEnum
from functools import cache
from itertools import permutations as perms
from itertools import product
from numbers import Integral
from types import MappingProxyType

import numpy as np


@dataclass(frozen=True, slots=True)
class _Tolerances:
    """Store numerical tolerances used by mesh-convention checks."""

    ref: float = 1.0e-12
    geom: float = 1.0e-12
    quad8_edge: float = 5.0e-2
    role_match: float = 3.5e-1


_TOL = _Tolerances()

# Fixed, irregular directions avoid rays aligned with the axis-aligned faces,
# edges and vertices common in FE meshes. The values are otherwise arbitrary;
# three well-separated directions provide a deterministic majority vote when
# one ray passes through a numerically ambiguous feature.
_POINT_IN_SURF_RAY_DIRECTS = np.array((
    (0.745, 0.371, 0.553),
    (-0.299, 0.877, 0.376),
    (0.461, -0.314, 0.830),
))


def _calc_ref_node_perm(
    ref: np.ndarray,
    transformed: np.ndarray,
) -> tuple[int, ...]:
    """Map transformed reference coordinates back to their source slots."""
    slots: list[int] = []
    for point in transformed:
        matches = np.flatnonzero(
            np.all(np.isclose(ref, point, atol=_TOL.ref), axis=1)
        )
        if matches.shape[0] != 1:
            raise ValueError(
                "Reference transformation does not preserve element roles."
            )
        slots.append(int(matches[0]))
    unique_slots = set(slots)
    if len(unique_slots) != ref.shape[0]:
        raise ValueError("Reference transformation is not a node permutation.")
    return tuple(slots)


def _check_mesh_2d(mesh_in: SimData) -> bool:
    """Return whether a mesh represents a two-dimensional topology."""
    if mesh_in.coords is None or mesh_in.connect is None:
        return False
    return not _check_vol_mesh(mesh_in)


def _check_vol_mesh(mesh_in: SimData) -> bool:
    """Return whether every connectivity table describes volume elements."""
    if mesh_in.coords is None or mesh_in.connect is None:
        return False

    num_coords = mesh_in.coords.shape[0]
    shift_all = _check_mesh_needs_zero_based_shift(mesh_in, num_coords)
    table_types: set[bool] = set()

    for name, connect_raw in mesh_in.connect.items():
        connect = _enforce_connect_arr_format(connect_raw, name)
        if _check_transpose_needed(connect, name, mesh_in):
            connect = connect.T

        if _check_table_needs_zero_based_shift(connect, num_coords, shift_all):
            connect = connect - 1

        if not _check_idxs_zero_based(connect, num_coords):
            raise ValueError(
                f"Connectivity table '{name}' has invalid indices."
            )

        table_types.add(
            _check_vol_connect_table(connect, mesh_in.coords)
        )

    if len(table_types) > 1:
        raise ValueError(
            "A SimData mesh cannot mix surface and volume connectivity tables."
        )

    return bool(table_types and table_types.pop())


def _check_vol_connect_table(
    connect: np.ndarray,
    coords: np.ndarray,
) -> bool:
    """Return whether a connectivity table represents volume elements."""
    nodes_per_elem = connect.shape[1]
    _validate_nodes_per_elem(nodes_per_elem)

    if nodes_per_elem in _VOL_ONLY_NODE_COUNTS:
        return True
    if nodes_per_elem in _SURF_ONLY_NODE_COUNTS:
        return False

    if _get_surf_spec(nodes_per_elem) is ELEMENT_SPECS[EElementType.QUAD8]:
        all_rows_are_quad8 = True
        for row in connect:
            if not _check_quad8_surf_row(row, coords):
                all_rows_are_quad8 = False
                break
        if all_rows_are_quad8:
            return False

    vol_spec = _get_vol_spec(nodes_per_elem)
    corner_idxs = np.asarray(vol_spec.corner_idxs, dtype=np.int64)
    for row in connect:
        cell_coords = coords[row[corner_idxs]]
        metric = _calc_vol_signed_metric(cell_coords)

        if abs(metric) > _TOL.geom:
            return True

    return False


def _check_quad8_surf_row(
    connect_row: np.ndarray,
    coords: np.ndarray,
) -> bool:
    """Return whether an eight-node row has the QUAD8 midside layout."""
    elem_coords = coords[connect_row]
    corners = elem_coords[:4]
    midsides = elem_coords[4:]
    edge_starts = corners
    edge_ends = np.roll(corners, -1, axis=0)
    edges = edge_ends - edge_starts
    edge_lengths = np.linalg.norm(edges, axis=1)
    if np.any(edge_lengths <= _TOL.geom):
        return False

    edge_params = np.sum((midsides - edge_starts) * edges, axis=1)
    edge_params /= edge_lengths**2
    closest = edge_starts + edge_params[:, None] * edges
    distances = np.linalg.norm(midsides - closest, axis=1)
    on_edges = distances <= _TOL.quad8_edge * edge_lengths
    between_corners = np.logical_and(edge_params >= 0.0, edge_params <= 1.0)
    return bool(np.all(np.logical_and(on_edges, between_corners)))


def _check_surf_connect_table(
    connect: np.ndarray,
    coords: np.ndarray,
    surf_only: bool = False,
) -> bool:
    """Return whether a connectivity table represents surface elements."""

    if surf_only:
        return True
    return not _check_vol_connect_table(connect, coords)


def _calc_surf_orient_flips(
    connect: np.ndarray,
    coords: np.ndarray,
) -> np.ndarray:
    """Return row reversals required by the std surface convention."""

    corner_idxs = _get_corner_idxs(connect.shape[1])
    representatives: dict[tuple[int, ...], int] = {}
    duplicate_of = np.arange(connect.shape[0], dtype=np.int64)
    for row_idx, row in enumerate(connect):
        key = tuple(sorted(int(node) for node in row[corner_idxs]))
        representative = representatives.setdefault(key, row_idx)
        duplicate_of[row_idx] = representative

    try:
        topology = _build_surf_topology(connect, np.unique(duplicate_of))
    except ValueError as error:
        if "Non-manifold surface edge" not in str(error):
            raise
        # Surface slices can deliberately contain non-manifold face sets. They
        # have no single orientable shell, so retain the historical per-face
        # behaviour rather than applying a false cavity interpretation.
        return _calc_indep_surf_orient_flips(connect, coords)

    flips = np.zeros(connect.shape[0], dtype=bool)
    closed_comps: list[
        tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]
    ] = []

    for rows, edge_keys in topology:
        rel = _calc_comp_rel_flips(rows, edge_keys)
        flips[rows] = rel
        oriented = _apply_surf_flips(connect[rows], rel)

        is_closed = all(len(edge_keys[key]) == 2 for key in edge_keys)
        if not is_closed:
            # An open non-planar sheet has no intrinsic exterior. Preserve a
            # coherent input orientation; planar sheets retain std CCW.
            comp_nodes = np.unique(oriented[:, _get_corner_idxs(
                oriented.shape[1]
            )])
            comp_coords = coords[comp_nodes]
            if _check_coplanar(comp_coords):
                metric = _calc_first_surf_metric(
                    oriented,
                    comp_coords,
                    coords,
                )
                if metric is not None and metric < 0.0:
                    flips[rows] = ~flips[rows]
            continue

        vol = _calc_surf_signed_vol(oriented, coords)
        if abs(vol) <= _TOL.geom:
            raise ValueError(
                "Closed surface component has zero signed volume; cannot "
                "select a material exterior."
            )
        if vol < 0.0:
            flips[rows] = ~flips[rows]
            oriented = _apply_surf_flips(
                oriented,
                np.ones(rows.shape[0], dtype=bool),
            )
        comp_points = coords[np.unique(oriented[:, corner_idxs])]
        closed_comps.append((
            rows,
            oriented,
            np.min(comp_points, axis=0),
            np.max(comp_points, axis=0),
        ))

    # A disconnected closed shell contained by another shell is a cavity. Its
    # material-outward normal must point into the void, so its signed volume is
    # negative after local edge consistency has been established.
    for comp_idx, comp in enumerate(closed_comps):
        rows, oriented, _, _ = comp
        point = _calc_comp_probe_point(oriented, coords)
        depth = 0
        for other_idx, other_comp in enumerate(closed_comps):
            _, other_oriented, bounds_min, bounds_max = other_comp
            if other_idx == comp_idx:
                continue
            if not np.all(point >= bounds_min - _TOL.geom):
                continue
            if not np.all(point <= bounds_max + _TOL.geom):
                continue
            depth += _check_point_in_closed_surf(
                point,
                other_oriented,
                coords,
            )
        if depth % 2:
            flips[rows] = ~flips[rows]

    for row_idx, representative in enumerate(duplicate_of):
        if row_idx == representative:
            continue
        same_orient = _check_surf_rows_same_orient(
            connect[row_idx],
            connect[representative],
            coords,
        )
        flips[row_idx] = flips[representative] ^ (not same_orient)

    return flips


def _calc_indep_surf_orient_flips(
    connect: np.ndarray,
    coords: np.ndarray,
) -> np.ndarray:
    """Calculate each surface row's orientation flip independently."""
    flips = np.zeros(connect.shape[0], dtype=bool)
    for row_idx, row in enumerate(connect):
        metric = _calc_winding_metric(row, coords, surf_only=True)
        flips[row_idx] = metric is not None and metric < 0.0
    return flips


def _build_surf_topology(
    connect: np.ndarray,
    active_rows: np.ndarray,
) -> list[tuple[np.ndarray, dict]]:
    """Build connected surface components keyed by their corner-node edges."""

    corner_idxs = _get_corner_idxs(connect.shape[1])
    edge_map: dict = {}
    for row_idx in active_rows:
        row = connect[row_idx]
        corners = row[corner_idxs]
        for node_a, node_b in zip(corners, np.roll(corners, -1)):
            directed = (int(node_a), int(node_b))
            key = tuple(sorted(directed))
            edge_map.setdefault(key, []).append((row_idx, directed))

    for key, uses in edge_map.items():
        if len(uses) > 2:
            raise ValueError(
                f"Non-manifold surface edge {key} has {len(uses)} "
                "incident faces."
            )

    neighbours: dict[int, set[int]] = {}
    for row in active_rows:
        neighbours[int(row)] = set()
    for uses in edge_map.values():
        if len(uses) == 2:
            row_a, _ = uses[0]
            row_b, _ = uses[1]
            neighbours[row_a].add(row_b)
            neighbours[row_b].add(row_a)

    comps: list[tuple[np.ndarray, dict]] = []
    unseen = {int(row) for row in active_rows}
    while unseen:
        seed = unseen.pop()
        rows = {seed}
        stack = [seed]
        while stack:
            row = stack.pop()
            for neighbour in neighbours[row]:
                if neighbour in unseen:
                    unseen.remove(neighbour)
                    rows.add(neighbour)
                    stack.append(neighbour)
        rows_arr = np.asarray(sorted(rows), dtype=np.int64)
        row_set = set(rows_arr.tolist())
        comp_edges = {}
        for key, uses in edge_map.items():
            if uses[0][0] in row_set:
                comp_edges[key] = uses
        comps.append((rows_arr, comp_edges))
    return comps


def _check_surf_rows_same_orient(
    row_a: np.ndarray,
    row_b: np.ndarray,
    coords: np.ndarray,
) -> bool:
    """Return whether duplicate surface rows have the same directed boundary."""

    corner_idxs = _get_corner_idxs(row_a.shape[0])
    corners_a = row_a[corner_idxs]
    corners_b = row_b[corner_idxs]
    edges_b = set()
    corners_b_next = np.roll(corners_b, -1)
    for node_a, node_b in zip(corners_b, corners_b_next):
        edges_b.add((int(node_a), int(node_b)))
    for node_a, node_b in zip(corners_a, np.roll(corners_a, -1)):
        if (int(node_a), int(node_b)) in edges_b:
            return True
        if (int(node_b), int(node_a)) in edges_b:
            return False
    raise ValueError("Duplicate surface faces do not share a boundary edge.")


def _calc_comp_rel_flips(
    rows: np.ndarray,
    edge_keys: dict,
) -> np.ndarray:
    """Find local reversals making shared edges traverse opposite ways."""

    row_set = set(rows.tolist())
    constraints: dict[int, list[tuple[int, bool]]] = {}
    for row in row_set:
        constraints[row] = []
    for uses in edge_keys.values():
        if len(uses) != 2:
            continue
        row_a, direct_a = uses[0]
        row_b, direct_b = uses[1]
        same_direct = direct_a == direct_b
        constraints[row_a].append((row_b, same_direct))
        constraints[row_b].append((row_a, same_direct))

    assigned: dict[int, bool] = {}
    for seed in rows:
        seed_int = int(seed)
        if seed_int in assigned:
            continue
        assigned[seed_int] = False
        stack = [seed_int]
        while stack:
            row = stack.pop()
            for neighbour, xor_flip in constraints[row]:
                expected = assigned[row] ^ xor_flip
                if neighbour in assigned:
                    if assigned[neighbour] != expected:
                        raise ValueError("Surface component is not orientable.")
                else:
                    assigned[neighbour] = expected
                    stack.append(neighbour)
    return np.asarray([assigned[int(row)] for row in rows], dtype=bool)


def _apply_surf_flips(connect: np.ndarray, flips: np.ndarray) -> np.ndarray:
    """Apply the requested winding reversal to each surface row."""
    out = np.array(connect, copy=True)
    for row_idx in np.flatnonzero(flips):
        out[row_idx] = _reverse_surf_row(out[row_idx])
    return out


def _calc_first_surf_metric(
    connect: np.ndarray,
    comp_coords: np.ndarray,
    coords: np.ndarray,
) -> float | None:
    """Calculate the first non-degenerate metric in a surface component."""
    normal = _calc_std_plane_normal(comp_coords)
    corner_idxs = _get_corner_idxs(connect.shape[1])
    for row in connect:
        metric = _calc_loc_polygon_signed_area(
            coords[row[corner_idxs]],
            normal,
        )
        if metric is not None and abs(metric) > _TOL.geom:
            return metric
    return None


def _calc_surf_signed_vol(
    connect: np.ndarray,
    coords: np.ndarray,
) -> float:
    """Calculate the signed volume enclosed by a triangulated surface."""
    corner_idxs = _get_corner_idxs(connect.shape[1])
    vol = 0.0
    for row in connect:
        points = coords[row[corner_idxs]]
        for point_idx in range(1, points.shape[0] - 1):
            vol += float(np.dot(
                points[0],
                np.cross(points[point_idx], points[point_idx + 1]),
            )) / 6.0
    return vol


def _calc_comp_probe_point(
    connect: np.ndarray,
    coords: np.ndarray,
) -> np.ndarray:
    """Calculate an interior probe point near a surface component."""
    corner_idxs = _get_corner_idxs(connect.shape[1])
    points = coords[connect[0, corner_idxs]]
    normal = np.cross(points[1] - points[0], points[2] - points[0])
    normal_norm = np.linalg.norm(normal)
    if normal_norm <= _TOL.geom:
        return np.mean(points, axis=0)
    extent = np.ptp(coords, axis=0)
    epsilon = max(float(np.linalg.norm(extent)) * 1.0e-9, _TOL.geom * 10.0)
    return np.mean(points, axis=0) + epsilon * normal / normal_norm


def _check_point_in_closed_surf(
    point: np.ndarray,
    connect: np.ndarray,
    coords: np.ndarray,
) -> bool:
    """Classify a point with parity ray casting against a closed surface."""

    corner_idxs = _get_corner_idxs(connect.shape[1])
    votes: list[bool] = []
    for direct in _POINT_IN_SURF_RAY_DIRECTS:
        direct = direct / np.linalg.norm(direct)
        hits = 0

        for row in connect:
            corners = coords[row[corner_idxs]]

            for point_idx in range(1, corners.shape[0] - 1):
                intersects = _check_ray_intersects_triangle(
                    point,
                    direct,
                    corners[0],
                    corners[point_idx],
                    corners[point_idx + 1],
                )
                if intersects:
                    hits += 1

        votes.append(bool(hits % 2))

    return sum(votes) >= 2


def _check_ray_intersects_triangle(
    origin: np.ndarray,
    direct: np.ndarray,
    point_a: np.ndarray,
    point_b: np.ndarray,
    point_c: np.ndarray,
) -> bool:
    """Return whether a forward ray intersects a triangle."""

    edge_ab = point_b - point_a
    edge_ac = point_c - point_a
    perpendicular = np.cross(direct, edge_ac)
    determinant = float(np.dot(edge_ab, perpendicular))
    if abs(determinant) <= _TOL.geom:
        return False

    inv_determinant = 1.0 / determinant
    offset = origin - point_a
    barycentric_u = inv_determinant * float(np.dot(offset, perpendicular))
    if barycentric_u <= _TOL.geom or barycentric_u >= 1.0 - _TOL.geom:
        return False

    cross_offset_edge = np.cross(offset, edge_ab)
    barycentric_v = inv_determinant * float(
        np.dot(direct, cross_offset_edge)
    )
    if (
        barycentric_v <= _TOL.geom
        or barycentric_u + barycentric_v >= 1.0 - _TOL.geom
    ):
        return False

    distance = inv_determinant * float(
        np.dot(edge_ac, cross_offset_edge)
    )

    return distance > _TOL.geom


def _enforce_surf_orient_table(
    connect: np.ndarray,
    coords: np.ndarray,
) -> np.ndarray:
    """Return a table with std surface-component orientations."""
    return _apply_surf_flips(
        connect, _calc_surf_orient_flips(connect, coords)
    )


def _copy_sim_data(
    mesh_in: SimData,
    connect: dict[str, np.ndarray] | None = None,
) -> SimData:
    """Copy simulation data, optionally replacing its connectivity."""
    mesh_out = SimData(
        mesh_type=mesh_in.mesh_type,
        time=mesh_in.time,
        coords=mesh_in.coords,
        connect=mesh_in.connect if connect is None else connect,
        side_sets=mesh_in.side_sets,
        node_vars=mesh_in.node_vars,
        elem_vars=mesh_in.elem_vars,
        glob_vars=mesh_in.glob_vars,
    )

    return mesh_out


def _check_perms_equiv(
    perm: tuple[int, ...],
    ref_perm: tuple[int, ...],
    symmetries: tuple[tuple[int, ...], ...],
) -> bool:
    """Return whether permutations differ only by a proper symmetry."""
    for symmetry in symmetries:

        transformed_slots: list[int] = []
        for slot in symmetry:
            transformed_slots.append(ref_perm[slot])
        transformed_perm = tuple(transformed_slots)

        if perm == transformed_perm:
            return True

    return False


def infer_mesh_convention(mesh_in: SimData) -> MeshConvention:
    """Infer one source-to-Riley ordering for each element family."""
    if mesh_in.coords is None or mesh_in.connect is None:
        raise MeshConvErr(
            "Mesh convention inference requires coordinates and connectivity."
        )

    inferred: dict[EElementType, tuple[int, ...]] = {}
    for name, connect_raw in mesh_in.connect.items():

        connect = _enforce_connect_arr_format(connect_raw, name)
        if _check_transpose_needed(connect, name, mesh_in):
            connect = connect.T
        if _check_zero_based_shift_needed(connect, mesh_in.coords.shape[0]):
            connect = connect - 1
        if not _check_idxs_zero_based(connect, mesh_in.coords.shape[0]):
            raise MeshConvErr(
                f"Connectivity table '{name}' has invalid indices."
            )

        nodes_per_elem = connect.shape[1]
        if nodes_per_elem in _SURF_ONLY_NODE_COUNTS:
            spec = _get_surf_spec(nodes_per_elem)
        elif nodes_per_elem in _VOL_ONLY_NODE_COUNTS:
            spec = _get_vol_spec(nodes_per_elem)
        elif _check_surf_mesh_type(mesh_in.mesh_type):
            spec = _get_surf_spec(nodes_per_elem)
        elif _check_vol_connect_table(connect, mesh_in.coords):
            spec = _get_vol_spec(nodes_per_elem)
        else:
            spec = _get_surf_spec(nodes_per_elem)

        elem_type = _get_elem_type_from_spec(spec)

        try:
            normalised = _enforce_node_order_from_geom(
                connect, mesh_in.coords, spec,
            )
        except ValueError as error:
            raise MeshConvErr(
                f"Could not infer '{name}' ({elem_type.value}); supply "
                "MeshConvention explicitly."
            ) from error

        perms: set[tuple[int, ...]] = set()
        for row, target in zip(connect, normalised, strict=True):
            perm_slots: list[int] = []
            for node_id in target:
                matching_slots = np.flatnonzero(row == node_id)
                perm_slots.append(int(matching_slots[0]))
            perms.add(tuple(perm_slots))

        representative = min(perms)
        symmetries = _get_elem_symmetries(elem_type)
        layouts_equiv = True

        for candidate_perm in perms:
            perms_equiv = _check_perms_equiv(
                candidate_perm,
                representative,
                symmetries,
            )
            if not perms_equiv:
                layouts_equiv = False
                break

        if not layouts_equiv:
            raise MeshConvErr(
                f"Connectivity table '{name}' contains multiple source "
                f"layouts for {elem_type.value}; supply MeshConvention."
            )

        perm = representative
        previous = inferred.get(elem_type)
        layouts_disagree = False

        if previous is not None:
            layouts_equiv = _check_perms_equiv(
                perm,
                previous,
                symmetries,
            )
            layouts_disagree = not layouts_equiv

        if layouts_disagree:
            raise MeshConvErr(
                
                    f"Multiple connectivity tables disagree on the "
                    f"{elem_type.value} source layout; supply "
                    "MeshConvention."
                
            )
        inferred[elem_type] = perm

    return MeshConvention(
        MappingProxyType(inferred),
        standardise_equiv_orients=True,
    )


def _check_surf_mesh_type(mesh_type: EMeshType | None) -> bool:
    """Return whether an explicit mesh type denotes a surface mesh."""
    return mesh_type is EMeshType.SURF


def _enforce_connect_arr_format(connect: np.ndarray, name: str) -> np.ndarray:
    """Return connectivity as a contiguous two-dimensional integer array."""
    arr = np.asarray(connect)
    if arr.ndim != 2:
        raise ValueError(
            f"Connectivity table '{name}' must be 2D, got shape {arr.shape}."
        )

    return np.ascontiguousarray(arr, dtype=np.int64)


def _validate_nodes_per_elem(nodes_per_elem: int) -> None:
    """Raise if an element node count is unsupported."""
    if nodes_per_elem not in _SUPPORTED_NODE_COUNTS:
        raise NotImplementedError(
            "Mesh convention tools do not support elements with "
            f"{nodes_per_elem} nodes."
        )


def _check_transpose_needed(
    connect: np.ndarray,
    connect_name: str | None = None,
    mesh_in: SimData | None = None,
) -> bool:
    """Return whether connectivity must be transposed to row-major form."""
    rows_supported = connect.shape[0] in _SUPPORTED_NODE_COUNTS
    cols_supported = connect.shape[1] in _SUPPORTED_NODE_COUNTS

    if rows_supported and not cols_supported:
        return True

    if cols_supported and not rows_supported:
        return False

    if cols_supported and rows_supported:
        mesh_is_vol = mesh_in is not None and mesh_in.mesh_type is EMeshType.VOL
        if mesh_is_vol:
            rows_vol_only = connect.shape[0] in _VOL_ONLY_NODE_COUNTS
            cols_vol_only = connect.shape[1] in _VOL_ONLY_NODE_COUNTS
            if rows_vol_only != cols_vol_only:
                return rows_vol_only

        mesh_is_surf = (
            mesh_in is not None
            and mesh_in.mesh_type is EMeshType.SURF
        )
        if mesh_is_surf:
            rows_surf_only = connect.shape[0] in _SURF_ONLY_NODE_COUNTS
            cols_surf_only = connect.shape[1] in _SURF_ONLY_NODE_COUNTS
            if rows_surf_only != cols_surf_only:
                return rows_surf_only

        block_rows = _get_elem_var_row_counts(connect_name, mesh_in)
        if block_rows is not None:
            row_match = connect.shape[0] in block_rows
            col_match = connect.shape[1] in block_rows
            if row_match and not col_match:
                return False
            if col_match and not row_match:
                return True

        if mesh_in is not None and mesh_in.coords is not None:
            return (
                _calc_orient_score(connect.T, mesh_in.coords)
                > _calc_orient_score(connect, mesh_in.coords)
            )

        return False

    raise NotImplementedError(
        "Could not infer connectivity orientation from shape "
        f"{connect.shape}. Expected a supported nodes-per-element dimension."
    )


def _check_zero_based_shift_needed(
    connect: np.ndarray,
    num_coords: int,
) -> bool:
    """Return whether connectivity is unambiguously one-based."""
    if connect.size == 0:
        return False

    if np.any(connect < 0):
        return False

    if np.any(connect == 0):
        return False

    return bool(np.any(connect >= num_coords))


def _check_ambiguously_positive(connect: np.ndarray, num_coords: int) -> bool:
    """Return whether positive indices could use either indexing convention."""
    if connect.size == 0:
        return False

    return bool(
        np.all(connect >= 1)
        and np.all(connect < num_coords)
        and not np.any(connect == 0)
    )


def _check_mesh_needs_zero_based_shift(
    mesh_in: SimData,
    num_coords: int,
) -> bool:
    """Return whether all connectivity tables require a one-based shift."""
    any_zero_based = False
    any_definitely_one_based = False

    for connect_name, connect_raw in mesh_in.connect.items():
        connect = np.asarray(connect_raw, dtype=np.int64)

        if _check_transpose_needed(connect, connect_name, mesh_in):
            connect = connect.T

        if np.any(connect == 0):
            any_zero_based = True

        if _check_zero_based_shift_needed(connect, num_coords):
            any_definitely_one_based = True

    if any_zero_based and any_definitely_one_based:
        raise ValueError(
            
                "Mixed zero-based and one-based connectivity tables "
                "detected in the same mesh."
            
        )

    return any_definitely_one_based and not any_zero_based


def _check_table_needs_zero_based_shift(
    connect: np.ndarray,
    num_coords: int,
    shift_all: bool,
) -> bool:
    """Return whether one connectivity table requires a one-based shift."""
    if _check_zero_based_shift_needed(connect, num_coords):
        return True
    return shift_all and _check_ambiguously_positive(connect, num_coords)


def _get_elem_var_row_counts(
    connect_name: str | None,
    mesh_in: SimData | None,
) -> set[int] | None:
    """Return element-variable row counts associated with a mesh block."""
    if connect_name is None or mesh_in is None or mesh_in.elem_vars is None:
        return None

    try:
        block_id = int(connect_name.replace("connect", ""))
    except ValueError:
        return None

    row_counts: set[int] = set()
    for (_field_name, field_block), values in mesh_in.elem_vars.items():
        if field_block == block_id:
            row_counts.add(values.shape[0])
    return row_counts or None


def _check_idxs_zero_based(connect: np.ndarray, num_coords: int) -> bool:
    """Return whether every index is valid zero-based connectivity."""
    if connect.size == 0:
        return True
    valid_mask = np.logical_and(connect >= 0, connect < num_coords)
    return bool(np.all(valid_mask))


def _calc_orient_score(
    connect_row_major: np.ndarray,
    coords: np.ndarray,
) -> int:
    """Score a candidate row-major interpretation by valid element rows."""
    num_coords = coords.shape[0]
    connect_eval = np.asarray(connect_row_major, dtype=np.int64)

    if (
        connect_eval.ndim != 2
        or connect_eval.shape[1] not in _SUPPORTED_NODE_COUNTS
    ):
        return -1

    if _check_zero_based_shift_needed(connect_eval, num_coords):
        connect_eval = connect_eval - 1

    if not _check_idxs_zero_based(connect_eval, num_coords):
        return -1

    score = 0
    for row in connect_eval:
        if np.unique(row).shape[0] != row.shape[0]:
            continue

        try:
            metric = _calc_row_orient_metric(row, coords)
        except ValueError:
            continue

        if metric is not None and abs(metric) > _TOL.geom:
            score += 1

    return score


def _calc_row_orient_metric(
    connect_row: np.ndarray,
    coords: np.ndarray,
) -> float | None:
    """Return a metric only for resolving row-major connectivity ambiguity."""
    nodes_per_elem = connect_row.shape[0]

    if nodes_per_elem in _SURF_NODE_COUNTS:
        corner_coords = coords[connect_row[_get_corner_idxs(nodes_per_elem)]]
        if _check_coplanar(corner_coords):
            return _calc_polygon_signed_area(corner_coords)

    if nodes_per_elem in _VOL_NODE_COUNTS:
        vol_spec = _get_vol_spec(nodes_per_elem)
        vol_coords = coords[
            connect_row[np.asarray(vol_spec.corner_idxs, dtype=np.int64)]
        ]
        metric = _calc_vol_signed_metric(vol_coords)
        if (
            nodes_per_elem in _VOL_ONLY_NODE_COUNTS
            or abs(metric) > _TOL.geom
        ):
            return metric

    return None


def _enforce_node_order_table(
    connect: np.ndarray,
    coords: np.ndarray,
    surf_only: bool = False,
    src_convention: MeshConvention | None = None,
) -> np.ndarray:
    """Rebuild higher-order rows from geometry, independent of source order."""
    if connect.shape[1] in _SURF_ONLY_NODE_COUNTS:
        spec = _get_surf_spec(connect.shape[1])
    elif connect.shape[1] in _VOL_ONLY_NODE_COUNTS:
        spec = _get_vol_spec(connect.shape[1])
    elif surf_only or not _check_vol_connect_table(connect, coords):
        spec = _get_surf_spec(connect.shape[1])
    else:
        spec = _get_vol_spec(connect.shape[1])

    if src_convention is not None:
        perm = src_convention.get_src_perm(
            _get_elem_type_from_spec(spec)
        )
        if perm is not None:
            perm_size_valid = len(perm) == connect.shape[1]
            expected_slots = set(range(connect.shape[1]))
            perm_slots_valid = set(perm) == expected_slots
            if not perm_size_valid or not perm_slots_valid:
                raise ValueError(
                    "MeshConvention permutation does not match the element "
                    f"topology {_get_elem_type_from_spec(spec).value}."
                )
            normalised = np.ascontiguousarray(
                connect[:, np.asarray(perm, dtype=np.int64)],
                dtype=np.int64,
            )
            if src_convention.standardise_equiv_orients:
                return _enforce_std_orients(normalised, spec)
            return normalised

    return connect


def _enforce_node_order_high_order(
    connect: np.ndarray,
    coords: np.ndarray,
    spec: ElementSpec,
) -> np.ndarray:
    """Rebuild higher-order rows from inferred geometric roles."""

    connect_out = np.empty_like(connect)
    for row_idx, row in enumerate(connect):
        connect_out[row_idx] = _enforce_node_order_high_order_row(
            row,
            coords,
            spec,
        )

    return np.ascontiguousarray(connect_out, dtype=np.int64)


def _enforce_node_order_from_geom(
    connect: np.ndarray,
    coords: np.ndarray,
    spec: ElementSpec,
) -> np.ndarray:
    """Infer node roles, then select the unique ID-stable proper orientation."""

    if connect.shape[1] == len(spec.corner_idxs):
        connect_out = np.empty_like(connect)
        src_slots = np.arange(connect.shape[1], dtype=np.int64)

        for row_idx, row in enumerate(connect):
            if spec.is_surf:
                ordered = _order_surf_corners(coords[row], src_slots)
            elif len(spec.corner_idxs) == 4:
                ordered = _order_tet_corners(coords[row], src_slots)
            else:
                ordered = _order_hex_corners(coords[row], src_slots)
            connect_out[row_idx] = row[ordered]
    else:
        connect_out = _enforce_node_order_high_order(connect, coords, spec)

    return _enforce_std_orients(connect_out, spec)


def _enforce_std_orients(
    connect: np.ndarray,
    spec: ElementSpec,
) -> np.ndarray:
    """Anchor each role-correct row at the lowest valid global-node sequence."""
    elem_type = _get_elem_type_from_spec(spec)
    perms_arr = _get_elem_symmetry_arrs(elem_type)
    out = np.empty_like(connect)
    for row_idx, row in enumerate(connect):
        out[row_idx] = min(
            (row[perm] for perm in perms_arr),
            key=lambda candidate: tuple(int(node) for node in candidate),
        )
    return np.ascontiguousarray(out, dtype=np.int64)


def _enforce_node_order_high_order_row(
    connect_row: np.ndarray,
    coords: np.ndarray,
    spec: ElementSpec,
) -> np.ndarray:
    """Infer std node roles for one higher-order element row."""
    elem_coords = coords[connect_row]
    if _check_std_node_roles(elem_coords, spec):
        return connect_row

    corner_count = len(spec.corner_idxs)
    corner_loc = _infer_corner_nodes(elem_coords, corner_count)

    if spec.is_surf:
        corner_loc = _order_surf_corners(elem_coords, corner_loc)
    elif corner_count == 4:
        corner_loc = _order_tet_corners(elem_coords, corner_loc)
    else:
        corner_loc = _order_hex_corners(elem_coords, corner_loc)

    corner_coords = elem_coords[corner_loc]
    remaining: list[int] = []
    for idx in range(connect_row.shape[0]):
        if idx not in corner_loc:
            remaining.append(idx)

    edge_pairs = spec.edge_pairs
    if not edge_pairs:
        edge_pairs_out: list[tuple[int, int]] = []
        for idx in range(corner_count):
            edge_pairs_out.append((idx, (idx + 1) % corner_count))
        edge_pairs = tuple(edge_pairs_out)
    edge_targets_out: list[np.ndarray] = []
    for start, end in edge_pairs:
        edge_targets_out.append(
            0.5 * (corner_coords[start] + corner_coords[end]),
        )
    edge_targets = np.asarray(edge_targets_out, dtype=np.float64)
    edge_loc = _match_role_nodes(
        elem_coords,
        remaining,
        edge_targets,
        "edge",
    )
    remaining_out = []
    for idx in remaining:
        if idx not in edge_loc:
            remaining_out.append(idx)
    remaining = remaining_out

    if spec.face_centre_idxs:
        face_targets_out: list[np.ndarray] = []
        for face in spec.face_corner_idxs:
            face_idxs = list(face)
            face_targets_out.append(
                np.mean(corner_coords[face_idxs], axis=0),
            )

        face_targets = np.asarray(face_targets_out, dtype=np.float64)
        face_loc = _match_role_nodes(
            elem_coords,
            remaining,
            face_targets,
            "face centre",
        )

        remaining_out = []
        for idx in remaining:
            if idx not in face_loc:
                remaining_out.append(idx)

        remaining = remaining_out

    if spec.centre_idx is not None or spec.cell_centre_idx is not None:
        centre_loc = _match_role_nodes(
            elem_coords,
            remaining,
            np.mean(corner_coords, axis=0, keepdims=True),
            "centre",
        )

        remaining_out = []
        for idx in remaining:
            if idx not in centre_loc:
                remaining_out.append(idx)

        remaining = remaining_out

    if remaining:
        raise ValueError(
            "Could not assign every higher-order node to an element role."
        )

    out_loc = np.empty(connect_row.shape[0], dtype=np.int64)
    out_loc[np.asarray(spec.corner_idxs, dtype=np.int64)] = corner_loc
    edge_slots = np.arange(
        corner_count, corner_count + len(edge_pairs), dtype=np.int64
    )
    out_loc[edge_slots] = edge_loc

    if spec.face_centre_idxs:
        out_loc[
            np.asarray(spec.face_centre_idxs, dtype=np.int64)
        ] = face_loc

    if spec.centre_idx is not None:
        out_loc[spec.centre_idx] = centre_loc[0]

    if spec.cell_centre_idx is not None:
        out_loc[spec.cell_centre_idx] = centre_loc[0]

    return connect_row[out_loc]


def _check_std_node_roles(
    elem_coords: np.ndarray,
    spec: ElementSpec,
) -> bool:
    """Accept an already coherent row, including curved and seam elements."""
    corner_idxs = np.asarray(spec.corner_idxs, dtype=np.int64)
    corner_coords = elem_coords[corner_idxs]

    rank_required = 2 if spec.is_surf else 3
    corner_center = np.mean(corner_coords, axis=0)
    centered_corners = corner_coords - corner_center
    corner_rank = np.linalg.matrix_rank(centered_corners, tol=_TOL.geom)
    if corner_rank < rank_required:
        return False

    inferred_corners = _infer_corner_nodes(elem_coords, len(corner_idxs))
    if (set(inferred_corners) == set(corner_idxs)
            and inferred_corners.shape[0] == len(corner_idxs)):
        pass
    elif inferred_corners.shape[0] == len(corner_idxs):
        return False

    scale = max(float(np.ptp(elem_coords, axis=0).max()), _TOL.geom)
    edge_pairs = spec.edge_pairs
    if not edge_pairs:

        edge_pairs_out = []
        for idx in range(len(corner_idxs)):
            edge_pairs_out.append((idx, (idx + 1) % len(corner_idxs)))

        edge_pairs = tuple(edge_pairs_out)

    edge_start = len(corner_idxs)
    edge_stop = edge_start + len(edge_pairs)
    edge_nodes = elem_coords[edge_start:edge_stop]
    edge_distances_out: list[float] = []

    for node, (start, end) in zip(edge_nodes, edge_pairs, strict=True):
        edge_distances_out.append(_calc_point_segment_distance(
            node,
            corner_coords[start],
            corner_coords[end],
        ))

    edge_distances = np.asarray(edge_distances_out)

    if np.any(edge_distances > _TOL.role_match * scale):
        return False

    if spec.face_centre_idxs:
        face_data = zip(
            spec.face_centre_idxs,
            spec.face_corner_idxs,
            strict=True,
        )
        for centre_idx, face_corner_idxs in face_data:
            face_coords = corner_coords[np.asarray(face_corner_idxs)]
            face_target = np.mean(face_coords, axis=0)
            face_error = np.linalg.norm(
                elem_coords[centre_idx] - face_target,
            )
            if face_error > _TOL.role_match * scale:
                return False

    if spec.centre_idx is not None:
        centre_error = np.linalg.norm(
            elem_coords[spec.centre_idx] - corner_center,
        )

        if centre_error > _TOL.role_match * scale:
            return False

    if spec.cell_centre_idx is not None:
        cell_centre_error = np.linalg.norm(
            elem_coords[spec.cell_centre_idx] - corner_center,
        )

        if cell_centre_error > _TOL.role_match * scale:
            return False

    return True


def _infer_corner_nodes(
    elem_coords: np.ndarray,
    corner_count: int,
) -> np.ndarray:
    """Find affine-element vertices without depending on their source slots."""

    scale = max(float(np.ptp(elem_coords, axis=0).max()), _TOL.geom)
    midpoint_tol = _TOL.geom * scale * 100.0
    midpoint_nodes: set[int] = set()

    for node_idx, point in enumerate(elem_coords):
        for first in range(elem_coords.shape[0]):
            for second in range(first + 1, elem_coords.shape[0]):

                if node_idx in (first, second):
                    continue

                midpoint = 0.5 * (elem_coords[first] + elem_coords[second])

                if np.linalg.norm(point - midpoint) <= midpoint_tol:
                    midpoint_nodes.add(node_idx)
                    break

            if node_idx in midpoint_nodes:
                break

    candidates_out: list[int] = []
    for idx in range(elem_coords.shape[0]):
        if idx not in midpoint_nodes:
            candidates_out.append(idx)

    candidates = np.asarray(candidates_out, dtype=np.int64)
    if candidates.shape[0] == corner_count:
        return candidates

    distances = np.linalg.norm(
        elem_coords - np.mean(elem_coords, axis=0),
        axis=1,
    )

    return np.sort(np.argsort(distances)[-corner_count:])


def _calc_point_segment_distance(
    point: np.ndarray,
    start: np.ndarray,
    end: np.ndarray,
) -> float:
    """Calculate the shortest distance from a point to a line segment."""
    direct = end - start
    length_sq = float(np.dot(direct, direct))

    if length_sq <= _TOL.geom:
        return np.inf

    param = np.clip(
        float(np.dot(point - start, direct) / length_sq), 0.0, 1.0
    )

    return float(np.linalg.norm(point - (start + param * direct)))


def _order_surf_corners(
    elem_coords: np.ndarray,
    corner_loc: np.ndarray,
) -> np.ndarray:
    """Return surface-corner slots in counter-clockwise order."""

    corner_coords = elem_coords[corner_loc]
    centred = corner_coords - np.mean(corner_coords, axis=0)
    if np.linalg.matrix_rank(centred, tol=_TOL.geom) < 2:
        raise ValueError(
            "Degenerate surface element has collinear corner nodes."
        )

    _, _, vecs = np.linalg.svd(centred, full_matrices=False)
    normal = vecs[-1]
    axis_u = vecs[0]
    axis_v = np.cross(normal, axis_u)
    angles = np.arctan2(centred @ axis_v, centred @ axis_u)

    ordered = corner_loc[np.argsort(angles)]
    ordered_coords = elem_coords[ordered]

    signed = np.dot(
        np.cross(
            ordered_coords[1] - ordered_coords[0],
            ordered_coords[2] - ordered_coords[0],
        ),
        normal,
    )

    if signed < 0.0:
        ordered = ordered[[0, *range(len(ordered) - 1, 0, -1)]]

    return ordered


def _order_tet_corners(
    elem_coords: np.ndarray,
    corner_loc: np.ndarray,
) -> np.ndarray:
    """Return tetrahedron-corner slots in right-handed order."""
    ordered = corner_loc[np.lexsort(elem_coords[corner_loc].T[::-1])]
    corner_coords = elem_coords[ordered]
    metric = _calc_tet_signed_vol(corner_coords)

    if abs(metric) <= _TOL.geom:
        raise ValueError("Degenerate tetrahedron has zero signed volume.")

    if metric < 0.0:
        ordered[[1, 2]] = ordered[[2, 1]]

    return ordered


def _order_hex_corners(
    elem_coords: np.ndarray,
    corner_loc: np.ndarray,
) -> np.ndarray:
    """Return hexahedron-corner slots in right-handed Riley order."""

    corner_coords = elem_coords[corner_loc]
    corner_center = np.mean(corner_coords, axis=0)
    centered_corners = corner_coords - corner_center
    corner_rank = np.linalg.matrix_rank(centered_corners, tol=_TOL.geom)

    if corner_rank < 3:
        raise ValueError("Degenerate hexahedron has coplanar corner nodes.")

    origin = corner_loc[np.lexsort(corner_coords.T[::-1])[0]]
    candidates = []
    for idx in corner_loc:
        if idx != origin:
            candidates.append(idx)

    best_order: np.ndarray | None = None
    best_error = np.inf
    origin_coord = elem_coords[origin]

    for first in candidates:
        for second in candidates:
            for third in candidates:
                if len({first, second, third}) != 3:
                    continue

                directs = np.array((
                    elem_coords[first] - origin_coord,
                    elem_coords[second] - origin_coord,
                    elem_coords[third] - origin_coord,
                ))

                determinant = np.linalg.det(directs)
                if determinant <= _TOL.geom:
                    continue

                targets = np.array((
                    origin_coord,
                    origin_coord + directs[0],
                    origin_coord + directs[0] + directs[1],
                    origin_coord + directs[1],
                    origin_coord + directs[2],
                    origin_coord + directs[0] + directs[2],
                    origin_coord
                    + directs[0]
                    + directs[1]
                    + directs[2],
                    origin_coord + directs[1] + directs[2],
                ))

                assigned = _match_role_nodes(
                    elem_coords,
                    list(corner_loc),
                    targets,
                    "corner",
                    validate=False,
                )

                error = float(np.sum(
                    np.linalg.norm(elem_coords[assigned] - targets, axis=1)
                ))

                if error < best_error:
                    best_error = error
                    best_order = np.asarray(assigned, dtype=np.int64)

    if best_order is None:
        raise ValueError(
            "Could not determine a right-handed hexahedron corner order."
        )

    scale = float(np.max(
        np.linalg.norm(corner_coords - origin_coord, axis=1)
    ))

    if best_error > _TOL.role_match * scale:
        raise ValueError("Hexahedron corner nodes do not form a valid element.")

    return best_order


def _match_role_nodes(
    elem_coords: np.ndarray,
    candidate_loc: list[int],
    targets: np.ndarray,
    role_name: str,
    validate: bool = True,
) -> list[int]:
    """Match unused local nodes to target geometric roles."""

    if len(candidate_loc) < targets.shape[0]:
        raise ValueError(
            f"Not enough nodes available to assign {role_name} roles."
        )

    candidates = np.asarray(candidate_loc, dtype=np.int64)
    distances = np.linalg.norm(
        elem_coords[candidates, None, :] - targets[None, :, :],
        axis=2,
    )

    assigned: list[int] = []
    used: set[int] = set()
    for target_idx in range(targets.shape[0]):
        ranked = np.argsort(distances[:, target_idx])
        match_idx = None

        for idx in ranked:
            if int(idx) not in used:
                match_idx = idx
                break

        if match_idx is None:
            raise ValueError(f"Could not assign a unique {role_name} node.")

        used.add(int(match_idx))
        assigned.append(int(candidates[match_idx]))

    if validate:
        scale = max(float(np.ptp(elem_coords, axis=0).max()), _TOL.geom)
        errors = np.linalg.norm(elem_coords[assigned] - targets, axis=1)
        if np.any(errors > _TOL.role_match * scale):
            raise ValueError(
                f"Could not match {role_name} nodes to the element geometry."
            )

    return assigned


def _calc_active_coord_axes(coords: np.ndarray) -> np.ndarray:
    """Return the first two coordinate axes with nonzero extent."""
    axis_range = np.ptp(coords, axis=0)
    active = np.flatnonzero(axis_range > _TOL.geom)

    if active.shape[0] < 2:
        raise ValueError("At least two active coordinate axes are required.")

    return active[:2]


def _calc_polygon_signed_area(coords_elem: np.ndarray) -> float:
    """Calculate polygon signed area on its active coordinate plane."""
    axes = _calc_active_coord_axes(coords_elem)
    xy = coords_elem[:, axes]
    rolled = np.roll(xy, -1, axis=0)
    return 0.5 * np.sum(xy[:, 0] * rolled[:, 1] - rolled[:, 0] * xy[:, 1])


def _check_coplanar(coords_elem: np.ndarray) -> bool:
    """Return whether coordinates lie on one plane."""
    centred = coords_elem - np.mean(coords_elem, axis=0)
    return np.linalg.matrix_rank(centred, tol=_TOL.geom) <= 2


def _calc_tet_signed_vol(coords_elem: np.ndarray) -> float:
    """Calculate a tetrahedron's signed volume metric."""
    return float(
        np.linalg.det(
            np.column_stack((
                coords_elem[1] - coords_elem[0],
                coords_elem[2] - coords_elem[0],
                coords_elem[3] - coords_elem[0],
            ))
        )
    )


def _calc_hex_signed_vol(coords_elem: np.ndarray) -> float:
    """Calculate a hexahedron's signed corner metric."""
    return float(
        np.linalg.det(
            np.column_stack((
                coords_elem[1] - coords_elem[0],
                coords_elem[3] - coords_elem[0],
                coords_elem[4] - coords_elem[0],
            ))
        )
    )


def _calc_vol_signed_metric(corner_coords: np.ndarray) -> float:
    """Calculate the signed metric for tetrahedral or hexahedral corners."""
    if corner_coords.shape[0] == 4:
        return _calc_tet_signed_vol(corner_coords)
    return _calc_hex_signed_vol(corner_coords)


def _calc_winding_metric(
    connect_row: np.ndarray,
    coords: np.ndarray,
    surf_only: bool = False,
) -> float | None:
    """Calculate a signed winding metric when a row is a surface element."""
    nodes_per_elem = connect_row.shape[0]
    if nodes_per_elem not in _SURF_NODE_COUNTS:
        return None

    if not surf_only and nodes_per_elem not in _SURF_ONLY_NODE_COUNTS:
        vol_spec = _get_vol_spec(nodes_per_elem)
        cell_coords = coords[
            connect_row[np.asarray(vol_spec.corner_idxs, dtype=np.int64)]
        ]
        signed_metric = _calc_vol_signed_metric(cell_coords)
        if abs(signed_metric) > _TOL.geom:
            return None

    corner_idxs = _get_corner_idxs(nodes_per_elem)
    coords_elem = coords[connect_row[corner_idxs]]

    if _check_coplanar(coords):
        ref_normal = _calc_std_plane_normal(coords)
    else:
        face_centroid = np.mean(coords_elem, axis=0)
        outward = face_centroid - np.mean(coords, axis=0)
        outward_norm = np.linalg.norm(outward)
        if outward_norm <= _TOL.geom:
            return None
        ref_normal = outward / outward_norm

    return _calc_loc_polygon_signed_area(coords_elem, ref_normal)


def _calc_std_plane_normal(coords: np.ndarray) -> np.ndarray:
    """Calculate a deterministically signed normal for planar coordinates."""
    centred = coords - np.mean(coords, axis=0)
    _, _, vecs = np.linalg.svd(centred, full_matrices=False)
    normal = vecs[-1]
    dominant_axis = int(np.argmax(np.abs(normal)))

    if normal[dominant_axis] < 0.0:
        normal = -normal

    return normal / np.linalg.norm(normal)


def _calc_loc_polygon_signed_area(
    coords_elem: np.ndarray,
    ref_normal: np.ndarray,
) -> float | None:
    """Calculate polygon area signed against a reference normal."""
    origin = coords_elem[0]
    axis_u = None
    for point in coords_elem[1:]:
        edge = point - origin
        edge -= np.dot(edge, ref_normal) * ref_normal
        edge_norm = np.linalg.norm(edge)
        if edge_norm > _TOL.geom:
            axis_u = edge / edge_norm
            break

    if axis_u is None:
        return None

    axis_v = np.cross(ref_normal, axis_u)
    projected = np.column_stack((
        (coords_elem - origin) @ axis_u,
        (coords_elem - origin) @ axis_v,
    ))
    rolled = np.roll(projected, -1, axis=0)

    return float(0.5 * np.sum(
        projected[:, 0] * rolled[:, 1]
        - rolled[:, 0] * projected[:, 1],
    ))


def _calc_handedness_metric(
    connect_row: np.ndarray,
    coords: np.ndarray,
    surf_only: bool = False,
) -> float | None:
    """Calculate the applicable signed handedness metric for an element."""
    nodes_per_elem = connect_row.shape[0]
    if surf_only:
        return _calc_winding_metric(connect_row, coords, surf_only=True)

    if nodes_per_elem in _VOL_NODE_COUNTS:
        vol_spec = _get_vol_spec(nodes_per_elem)
        vol_coords = coords[
            connect_row[np.asarray(vol_spec.corner_idxs, dtype=np.int64)]
        ]
        vol_metric = _calc_vol_signed_metric(vol_coords)

        if nodes_per_elem in _VOL_ONLY_NODE_COUNTS:
            return vol_metric

        if abs(vol_metric) > _TOL.geom:
            return vol_metric

    if nodes_per_elem in _SURF_NODE_COUNTS:
        metric = _calc_winding_metric(connect_row, coords)

        if metric is not None:
            return metric

    raise NotImplementedError(
        
            f"Handedness checks are not implemented for "
            f"{nodes_per_elem}-node elements."
        
    )


def _check_ccw_winding_table(
    connect: np.ndarray,
    coords: np.ndarray,
    surf_only: bool = False,
) -> bool:
    """Return whether all applicable rows have counter-clockwise winding."""
    for row in connect:
        metric = _calc_winding_metric(row, coords, surf_only=surf_only)

        if metric is None:
            continue

        if abs(metric) <= _TOL.geom:
            continue

        if metric <= 0.0:
            return False

    return True


def _check_right_handed_table(
    connect: np.ndarray,
    coords: np.ndarray,
    surf_only: bool = False,
) -> bool:
    """Return whether all applicable rows have positive handedness."""
    for row in connect:
        metric = _calc_handedness_metric(row, coords, surf_only=surf_only)

        if metric is None:
            continue

        if abs(metric) <= _TOL.geom:
            continue

        if metric <= 0.0:
            return False

    return True


def _reverse_surf_row(connect_row: np.ndarray) -> np.ndarray:
    """Reverse winding while preserving every surface-node role."""

    spec = _get_surf_spec(connect_row.shape[0])
    if spec.surf_reverse_perm is None:
        raise NotImplementedError(
            
                f"Surface reversal is not implemented for "
                f"{spec.nodes_per_elem}-node elements."
            
        )

    perm = np.asarray(
        spec.surf_reverse_perm,
        dtype=np.int64,
    )

    return connect_row[perm]


def _reverse_handedness_row(connect_row: np.ndarray) -> np.ndarray:
    """Reverse handedness while preserving every volume-node role."""
    spec = _get_vol_spec(connect_row.shape[0])

    if spec.handedness_reverse_perm is None:
        raise NotImplementedError(
        
            f"Handedness reversal is not implemented for "
            f"{spec.nodes_per_elem}-node elements."
        
    )

    perm = np.asarray(
        spec.handedness_reverse_perm,
        dtype=np.int64,
    )

    return connect_row[perm]


def _get_face_corner_coords(face_coords: np.ndarray) -> np.ndarray:
    """Return only the corner coordinates from a surface face."""
    nodes_per_face = face_coords.shape[0]
    corner_idxs = np.asarray(
        _get_surf_spec(nodes_per_face).corner_idxs,
        dtype=np.int64,
    )

    return face_coords[corner_idxs]


def _calc_face_normal(face_coords: np.ndarray) -> np.ndarray:
    """Calculate a unit normal from a non-degenerate surface face."""
    face_corners = _get_face_corner_coords(face_coords)
    face_normal = np.cross(
        face_corners[1] - face_corners[0],
        face_corners[2] - face_corners[0],
    )
    normal_mag = np.linalg.norm(face_normal)

    if normal_mag <= _TOL.geom and face_corners.shape[0] == 4:
        face_normal = np.cross(
            face_corners[2] - face_corners[0],
            face_corners[3] - face_corners[0],
        )
        normal_mag = np.linalg.norm(face_normal)

    if normal_mag <= _TOL.geom:
        raise ValueError(
            "Degenerate face detected while extracting the surface mesh."
        )

    return face_normal / normal_mag


def _enforce_surf_face_outward(
    face_connect: np.ndarray,
    parent_connect: np.ndarray,
    coords: np.ndarray,
) -> np.ndarray:
    """Orient an extracted face away from its parent element."""
    face_coords = coords[face_connect]
    face_centroid = np.mean(_get_face_corner_coords(face_coords), axis=0)

    parent_corners = _get_vol_corner_idxs(parent_connect.shape[0])
    parent_centroid = np.mean(coords[parent_connect[parent_corners]], axis=0)

    face_normal = _calc_face_normal(face_coords)
    outward_dir = face_centroid - parent_centroid

    if np.dot(face_normal, outward_dir) < 0.0:
        return _reverse_surf_row(face_connect)

    return face_connect


def _enforce_surf_face_node_order(
    face_connect: np.ndarray,
    coords: np.ndarray,
) -> np.ndarray:
    """Restore high-order node roles on an extracted surface face."""
    nodes_per_face = face_connect.shape[0]
    face_out = np.copy(face_connect)
    face_coords = coords[face_out]

    spec = _get_surf_spec(nodes_per_face)
    corner_idxs = np.asarray(spec.corner_idxs, dtype=np.int64)
    num_corners = corner_idxs.shape[0]
    if nodes_per_face == num_corners:
        return face_out

    midside_pool = np.arange(num_corners, nodes_per_face, dtype=np.int64)
    edge_corner_pairs_out: list[tuple[int, int]] = []
    for corner_idx in range(num_corners):
        edge_corner_pairs_out.append(
            (corner_idx, (corner_idx + 1) % num_corners),
        )

    edge_corner_pairs = tuple(edge_corner_pairs_out)

    face_centroid = np.mean(face_coords[corner_idxs, :], axis=0)
    mid_pool_coords = face_coords[midside_pool, :]

    if spec.centre_idx is not None:
        centroid_dists = np.linalg.norm(mid_pool_coords - face_centroid, axis=1)
        center_pool_idx = int(np.argmin(centroid_dists))
        center_loc_idx = int(midside_pool[center_pool_idx])
        edge_pool_mask = np.ones(midside_pool.shape[0], dtype=bool)
        edge_pool_mask[center_pool_idx] = False
        edge_pool_loc_idxs = midside_pool[edge_pool_mask]
        edge_pool_coords = face_coords[edge_pool_loc_idxs, :]
    else:
        center_loc_idx = -1
        edge_pool_loc_idxs = midside_pool
        edge_pool_coords = mid_pool_coords

    edge_midpoints_out: list[np.ndarray] = []
    for start_idx, end_idx in edge_corner_pairs:
        edge_midpoints_out.append(
            0.5 * (
                face_coords[start_idx, :]
                + face_coords[end_idx, :]
            ),
        )

    edge_midpoints = np.asarray(edge_midpoints_out, dtype=np.float64)
    edge_dists = np.linalg.norm(
        edge_pool_coords[:, None, :] - edge_midpoints[None, :, :],
        axis=2,
    )
    edge_order = np.argmin(edge_dists, axis=0)
    reordered_edge_idxs = edge_pool_loc_idxs[edge_order]

    face_out[
        num_corners:num_corners + num_corners
    ] = face_out[reordered_edge_idxs]

    if spec.centre_idx is not None:
        face_out[spec.centre_idx] = face_out[center_loc_idx]

    return face_out


def _enforce_ccw_winding_table(
    connect: np.ndarray,
    coords: np.ndarray,
    surf_only: bool = False,
) -> np.ndarray:
    """Return connectivity with applicable rows wound counter-clockwise."""

    connect_out = np.copy(connect)
    for idx, row in enumerate(connect_out):
        metric = _calc_winding_metric(row, coords, surf_only=surf_only)

        if metric is not None and metric < 0.0:
            connect_out[idx, :] = _reverse_surf_row(row)

    return np.ascontiguousarray(connect_out, dtype=np.int64)


def _enforce_right_handed_table(
    connect: np.ndarray,
    coords: np.ndarray,
    surf_only: bool = False,
) -> np.ndarray:
    """Return connectivity with positive handedness where applicable."""

    connect_out = np.copy(connect)
    for idx, row in enumerate(connect_out):

        metric = _calc_handedness_metric(row, coords, surf_only=surf_only)
        if metric is not None and metric < 0.0:
            row_is_surf = row.shape[0] in _SURF_NODE_COUNTS
            row_is_coplanar = False

            if row_is_surf:
                corner_idxs = _get_corner_idxs(row.shape[0])
                corner_coords = coords[row[corner_idxs]]
                row_is_coplanar = _check_coplanar(corner_coords)

            reverse_surf = surf_only or (row_is_surf and row_is_coplanar)

            if reverse_surf:
                connect_out[idx, :] = _reverse_surf_row(row)
            else:
                connect_out[idx, :] = _reverse_handedness_row(row)

    return np.ascontiguousarray(connect_out, dtype=np.int64)


def _extract_surf_faces_from_table(
    connect: np.ndarray,
    coords: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Extract boundary faces and their parent rows from a volume table."""
    nodes_per_elem = connect.shape[1]
    face_map = _get_surf_map(nodes_per_elem)
    faces_wound = connect[:, face_map]
    faces_flat_wound = faces_wound.reshape((-1, face_map.shape[1]))
    faces_flat_sorted = np.sort(faces_flat_wound, axis=1)

    (_, unique_idxs, unique_counts) = np.unique(
        faces_flat_sorted,
        axis=0,
        return_index=True,
        return_counts=True,
    )

    ext_face_idxs = unique_idxs[unique_counts == 1]
    ext_parent_elem_idxs = np.ascontiguousarray(
        ext_face_idxs // face_map.shape[0],
        dtype=np.int64,
    )
    ext_faces = np.copy(faces_flat_wound[ext_face_idxs])

    for ff, parent_elem_idx in enumerate(ext_parent_elem_idxs):
        ext_faces[ff, :] = _enforce_surf_face_outward(
            ext_faces[ff, :],
            connect[parent_elem_idx, :],
            coords,
        )

        ext_faces[ff, :] = _enforce_surf_face_node_order(
            ext_faces[ff, :],
            coords,
        )

    ext_faces = np.ascontiguousarray(ext_faces, dtype=np.int64)

    return ext_faces, ext_parent_elem_idxs


def _restore_src_connect_style(
    mesh_in: SimData,
    one_based: bool,
    transposed: bool,
) -> SimData:
    """Restore the source indexing and table-orientation style."""
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


def _conv_to_vec3(
    values: np.ndarray | list[float] | tuple[float, ...],
    name: str,
) -> np.ndarray:
    """Return the first three finite components of an array-like value."""
    values_arr = np.asarray(values, dtype=np.float64).reshape(-1)

    if values_arr.size < 3:
        raise ValueError(f"'{name}' must contain at least three components.")

    vec = np.ascontiguousarray(values_arr[:3])
    finite = np.isfinite(vec)

    if not np.all(finite):
        raise ValueError(f"'{name}' must contain only finite values.")

    return vec


class MeshCheckCode(StrEnum):
    """A single mesh-convention condition that a connectivity table failed."""

    ROW_MAJOR_CONNECTIVITY = "row_major_connectivity"
    ZERO_BASED_INDEXING = "zero_based_indexing"
    CONNECTIVITY_INDICES = "connectivity_indices"
    CCW_WINDING = "ccw_winding"
    RIGHT_HANDED_GEOMETRY = "right_handed_geometry"
    SURFACE_TOPOLOGY = "surface_topology"
    NODE_ORDER = "node_order"


MeshConvCheck = dict[str, list[MeshCheckCode]]


class EMeshType(Enum):
    """Topological class of a Riley simulation mesh."""

    VOL = "volume"
    SURF = "surface"


class EElementType(Enum):
    """Supported finite-element topologies."""

    TRI3 = "tri3"
    TRI6 = "tri6"
    TRI7 = "tri7"
    QUAD4 = "quad4"
    QUAD8 = "quad8"
    QUAD9 = "quad9"
    TET4 = "tet4"
    TET10 = "tet10"
    HEX8 = "hex8"
    HEX20 = "hex20"
    HEX27 = "hex27"

    def calc_ref_coords(self) -> np.ndarray:
        """Return reference-node coordinates in Riley slot order.

        These coordinates are topology data, not geometry used to inspect a
        user mesh.  They let us derive every orientation-preserving
        automorphism of an element once, instead of maintaining hand-written
        permutations in several places.
        """
        if self in (EElementType.TRI3, EElementType.TRI6,
                    EElementType.TRI7):

            points = ((0., 0.), (1., 0.), (0., 1.), (.5, 0.), (.5, .5),
                      (0., .5))

            if self is EElementType.TRI3:
                points = points[:3]
            elif self is EElementType.TRI7:
                points += ((1. / 3., 1. / 3.),)

            return np.asarray(points, dtype=np.float64)

        if self in (EElementType.QUAD4, EElementType.QUAD8,
                    EElementType.QUAD9):

            points = ((0., 0.), (1., 0.), (1., 1.), (0., 1.),
                      (.5, 0.), (1., .5), (.5, 1.), (0., .5))

            if self is EElementType.QUAD4:
                points = points[:4]
            elif self is EElementType.QUAD9:
                points += ((.5, .5),)

            return np.asarray(points, dtype=np.float64)

        if self in (EElementType.TET4, EElementType.TET10):
            points = ((0., 0., 0.), (1., 0., 0.), (0., 1., 0.), (0., 0., 1.),
                      (.5, 0., 0.), (.5, .5, 0.), (0., .5, 0.),
                      (0., 0., .5), (.5, 0., .5), (0., .5, .5))

            return np.asarray(
                points[:4] if self is EElementType.TET4 else points,
                dtype=np.float64,
            )

        if self in (EElementType.HEX8, EElementType.HEX20,
                    EElementType.HEX27):
            points = ((0., 0., 0.), (1., 0., 0.), (1., 1., 0.), (0., 1., 0.),
                      (0., 0., 1.), (1., 0., 1.), (1., 1., 1.), (0., 1., 1.),
                      (.5, 0., 0.), (1., .5, 0.), (.5, 1., 0.), (0., .5, 0.),
                      (.5, 0., 1.), (1., .5, 1.), (.5, 1., 1.), (0., .5, 1.),
                      (0., 0., .5), (1., 0., .5), (1., 1., .5), (0., 1., .5),
                      (.5, 0., .5), (1., .5, .5), (.5, 1., .5), (0., .5, .5),
                      (.5, .5, 0.), (.5, .5, 1.), (.5, .5, .5))

            count = ELEMENT_SPECS[self].nodes_per_elem

            if self is EElementType.HEX20:
                return np.asarray(points[:20], dtype=np.float64)

            return np.asarray(points[:count], dtype=np.float64)

        raise ValueError(f"No reference coordinates for {self.value}.")

    def calc_orient_preserving_perms(
        self,
    ) -> tuple[tuple[int, ...], ...]:
        """Derive all proper topology symmetries in std-slot notation.

        A returned permutation maps a target Riley slot to a source Riley
        slot.  Applying one preserves all corner, edge, face-centre and
        volume-centre roles.  Surface reflections and volume inversions are
        deliberately absent.
        """
        spec = ELEMENT_SPECS[self]
        ref = self.calc_ref_coords()
        dims = 2 if spec.is_surf else 3
        corners = np.asarray(spec.corner_idxs, dtype=np.int64)
        candidates: tuple[tuple[int, ...], ...]

        if len(corners) == 8:
            # The proper rotational group of a cube: 3! axis orderings and sign
            # changes with positive determinant, for 24 transformations.
            candidate_rows: list[tuple[int, ...]] = []

            for axes in perms(range(3)):
                inversion_count = 0
                for idx in range(3):
                    for next_idx in range(idx + 1, 3):
                        inversion_count += axes[idx] > axes[next_idx]

                parity = 1 if inversion_count % 2 == 0 else -1

                for signs in product((-1., 1.), repeat=3):
                    sign_product = int(np.prod(signs))
                    if parity * sign_product < 0:
                        continue
                    transformed = (
                        (2.0 * ref - 1.0)[:, axes]
                        * np.asarray(signs)
                    )

                    transformed = 0.5 * (transformed + 1.0)
                    candidate_rows.append(
                        _calc_ref_node_perm(ref, transformed)
                    )

            candidates = tuple(candidate_rows)

        else:
            rows: list[tuple[int, ...]] = []
            src_corners = ref[corners, :dims]
            homogeneous = np.column_stack((src_corners,
                                           np.ones(len(corners))))

            for corner_perm in perms(range(len(corners))):
                target_corners = src_corners[np.asarray(corner_perm)]
                transform, _, _, _ = np.linalg.lstsq(
                    homogeneous,
                    target_corners,
                    rcond=None,
                )

                if np.linalg.det(transform[:dims]) <= 0.0:
                    continue

                transformed = np.column_stack(
                    (ref[:, :dims], np.ones(ref.shape[0]))
                ) @ transform

                try:
                    rows.append(_calc_ref_node_perm(
                        ref[:, :dims],
                        transformed,
                    ))
                except ValueError:
                    continue

            candidates = tuple(rows)

        return tuple(sorted(set(candidates)))


@dataclass(frozen=True, slots=True)
class MeshConvention:
    """Caller-declared source ordering for otherwise ambiguous elements.

    Each permutation maps a Riley std slot to the corresponding slot in
    the source connectivity row. Omitted element types are assumed to already
    use Riley ordering.
    """

    src_to_riley_perms: Mapping[EElementType, tuple[int, ...]]
    standardise_equiv_orients: bool = False

    def __post_init__(self) -> None:
        """Validate permutations and retain an immutable defensive copy."""
        perms_out: dict[EElementType, tuple[int, ...]] = {}
        for elem_type, perm_raw in (
            self.src_to_riley_perms.items()
        ):
            if not isinstance(elem_type, EElementType):
                raise TypeError(
                    "MeshConvention keys must be EElementType members."
                )
            perm = tuple(perm_raw)
            node_count = elem_type.calc_ref_coords().shape[0]
            if len(perm) != node_count:
                raise ValueError(
                    f"{elem_type.value} requires a {node_count}-slot "
                    "permutation."
                )
            slots_are_ints = True
            for slot in perm:
                if not isinstance(slot, Integral):
                    slots_are_ints = False
                    break
            if not slots_are_ints:
                raise TypeError("MeshConvention slots must be integers.")
            perm = tuple(int(slot) for slot in perm)
            perm_slots = set(perm)
            expected_slots = set(range(node_count))
            if perm_slots != expected_slots:
                raise ValueError(
                    f"{elem_type.value} perm must contain every "
                    f"slot from 0 to {node_count - 1} exactly once."
                )
            perms_out[elem_type] = perm

        object.__setattr__(
            self,
            "src_to_riley_perms",
            MappingProxyType(perms_out),
        )

    def get_src_perm(
        self,
        elem_type: EElementType,
    ) -> tuple[int, ...] | None:
        """Return the declared source permutation for an element type."""
        return self.src_to_riley_perms.get(elem_type)


class MeshConvErr(ValueError):
    """Raised when source node roles cannot be inferred unambiguously."""


@dataclass(frozen=True, slots=True)
class ElementSpec:
    """Static node-ordering metadata for one supported element topology."""

    nodes_per_elem: int
    is_surf: bool
    corner_idxs: tuple[int, ...]
    surf_reverse_perm: tuple[int, ...] | None = None
    handedness_reverse_perm: tuple[int, ...] | None = None
    surf_faces: tuple[tuple[int, ...], ...] | None = None
    centre_idx: int | None = None
    edge_pairs: tuple[tuple[int, int], ...] = ()
    face_corner_idxs: tuple[tuple[int, ...], ...] = ()
    face_centre_idxs: tuple[int, ...] = ()
    cell_centre_idx: int | None = None


@dataclass(slots=True)
class SimData:
    """Mesh data used by Riley's mesh-convention tools.

    Connectivity tables may initially use either indexing convention and either
    table orientation; :func:`enforce_mesh_convention` standardises them.
    The optional metadata fields allow external adapters to preserve associated
    data without making Riley depend on their data-model types.
    """

    coords: np.ndarray | None = None
    connect: dict[str, np.ndarray] | None = None
    mesh_type: EMeshType | None = None
    time: np.ndarray | None = None
    side_sets: dict[tuple[str, str], np.ndarray] | None = None
    node_vars: dict[str, np.ndarray] | None = None
    elem_vars: dict[tuple[str, int], np.ndarray] | None = None
    glob_vars: dict[str, np.ndarray] | None = None

    def update_mesh_type(self) -> None:
        """Update the mesh topology from coordinates and connectivity."""
        if self.coords is None or self.connect is None:
            self.mesh_type = None
            return
        if _check_vol_mesh(self):
            self.mesh_type = EMeshType.VOL
        else:
            self.mesh_type = EMeshType.SURF


ELEMENT_SPECS = MappingProxyType({
    EElementType.TRI3: ElementSpec(3, True, (0, 1, 2), (0, 2, 1)),
    EElementType.TRI6: ElementSpec(6, True, (0, 1, 2), (0, 2, 1, 5, 4, 3)),
    EElementType.TRI7: ElementSpec(
        7, True, (0, 1, 2), (0, 2, 1, 5, 4, 3, 6), centre_idx=6,
    ),
    EElementType.QUAD4: ElementSpec(4, True, (0, 1, 2, 3), (0, 3, 2, 1)),
    EElementType.QUAD8: ElementSpec(
        8,
        True,
        (0, 1, 2, 3),
        (0, 3, 2, 1, 7, 6, 5, 4),
    ),
    EElementType.QUAD9: ElementSpec(
        9, True, (0, 1, 2, 3), (0, 3, 2, 1, 7, 6, 5, 4, 8), centre_idx=8,
    ),
    EElementType.TET4: ElementSpec(
        4, False, (0, 1, 2, 3), None, (0, 2, 1, 3),
        ((0, 1, 2), (0, 3, 1), (0, 2, 3), (1, 3, 2)),
        edge_pairs=((0, 1), (1, 2), (2, 0), (0, 3), (1, 3), (2, 3)),
    ),
    EElementType.TET10: ElementSpec(
        10, False, (0, 1, 2, 3), None, (0, 2, 1, 3, 6, 5, 4, 7, 9, 8),
        ((0, 1, 2, 4, 5, 6), (0, 3, 1, 7, 8, 4),
         (0, 2, 3, 6, 9, 7), (1, 3, 2, 8, 9, 5)),
        edge_pairs=((0, 1), (1, 2), (2, 0), (0, 3), (1, 3), (2, 3)),
    ),
    EElementType.HEX8: ElementSpec(
        8, False, (0, 1, 2, 3, 4, 5, 6, 7), None, (0, 3, 2, 1, 4, 7, 6, 5),
        ((0, 1, 2, 3), (0, 3, 7, 4), (4, 7, 6, 5), (1, 5, 6, 2),
         (0, 4, 5, 1), (2, 6, 7, 3)),
        edge_pairs=((0, 1), (1, 2), (2, 3), (3, 0), (4, 5), (5, 6),
                    (6, 7), (7, 4), (0, 4), (1, 5), (2, 6), (3, 7)),
    ),
    EElementType.HEX20: ElementSpec(
        20, False, (0, 1, 2, 3, 4, 5, 6, 7), None,
        (0, 3, 2, 1, 4, 7, 6, 5, 11, 10, 9, 8, 15, 14, 13, 12, 16, 19, 18, 17),
        ((0, 1, 2, 3, 8, 9, 10, 11), (0, 3, 7, 4, 11, 15, 19, 16),
         (4, 7, 6, 5, 15, 14, 13, 12), (1, 5, 6, 2, 17, 13, 18, 9),
         (0, 4, 5, 1, 16, 12, 17, 8), (2, 6, 7, 3, 18, 14, 19, 10)),
        edge_pairs=((0, 1), (1, 2), (2, 3), (3, 0), (4, 5), (5, 6),
                    (6, 7), (7, 4), (0, 4), (1, 5), (2, 6), (3, 7)),
    ),
    EElementType.HEX27: ElementSpec(
        27, False, (0, 1, 2, 3, 4, 5, 6, 7), None,
        (0, 3, 2, 1, 4, 7, 6, 5, 11, 10, 9, 8, 15, 14, 13, 12, 16, 19, 18, 17,
         20, 21, 22, 23, 24, 25, 26),
        ((0, 1, 2, 3, 8, 9, 10, 11, 24), (0, 3, 7, 4, 11, 15, 19, 16, 23),
         (4, 7, 6, 5, 15, 14, 13, 12, 25), (1, 5, 6, 2, 17, 13, 18, 9, 21),
         (0, 4, 5, 1, 16, 12, 17, 8, 20), (2, 6, 7, 3, 18, 14, 19, 10, 22)),
        edge_pairs=((0, 1), (1, 2), (2, 3), (3, 0), (4, 5), (5, 6),
                    (6, 7), (7, 4), (0, 4), (1, 5), (2, 6), (3, 7)),
        face_corner_idxs=((0, 1, 2, 3), (0, 3, 7, 4), (4, 7, 6, 5),
                             (1, 5, 6, 2), (0, 4, 5, 1), (2, 6, 7, 3)),
        face_centre_idxs=(24, 23, 25, 21, 20, 22),
        cell_centre_idx=26,
    ),
})


_elem_specs_by_node_count_out: dict[int, tuple[ElementSpec, ...]] = {}
_supported_node_counts_out: set[int] = set()
for _spec in ELEMENT_SPECS.values():
    _supported_node_counts_out.add(_spec.nodes_per_elem)
for _nodes_per_elem in _supported_node_counts_out:
    _matching_specs: list[ElementSpec] = []
    for _spec in ELEMENT_SPECS.values():
        if _spec.nodes_per_elem == _nodes_per_elem:
            _matching_specs.append(_spec)
    _elem_specs_by_node_count_out[_nodes_per_elem] = tuple(_matching_specs)

_ELEM_SPECS_BY_NODE_COUNT = MappingProxyType(_elem_specs_by_node_count_out)


_surf_node_counts_out: set[int] = set()
for _spec in ELEMENT_SPECS.values():
    if _spec.is_surf:
        _surf_node_counts_out.add(_spec.nodes_per_elem)

_SURF_NODE_COUNTS = frozenset(_surf_node_counts_out)


_vol_node_counts_out: set[int] = set()
for _spec in ELEMENT_SPECS.values():
    if not _spec.is_surf:
        _vol_node_counts_out.add(_spec.nodes_per_elem)

_VOL_NODE_COUNTS = frozenset(_vol_node_counts_out)
_SUPPORTED_NODE_COUNTS = _SURF_NODE_COUNTS | _VOL_NODE_COUNTS


_surf_only_node_counts_out: set[int] = set()
for _nodes_per_elem, _specs in _ELEM_SPECS_BY_NODE_COUNT.items():
    _all_surf = True
    for _spec in _specs:
        if not _spec.is_surf:
            _all_surf = False
            break
    if _all_surf:
        _surf_only_node_counts_out.add(_nodes_per_elem)

_SURF_ONLY_NODE_COUNTS = frozenset(_surf_only_node_counts_out)


_vol_only_node_counts_out: set[int] = set()
for _nodes_per_elem, _specs in _ELEM_SPECS_BY_NODE_COUNT.items():
    _all_vol = True
    for _spec in _specs:
        if _spec.is_surf:
            _all_vol = False
            break
    if _all_vol:
        _vol_only_node_counts_out.add(_nodes_per_elem)

_VOL_ONLY_NODE_COUNTS = frozenset(_vol_only_node_counts_out)


@cache
def _get_surf_spec(nodes_per_elem: int) -> ElementSpec:
    """Return surface-element metadata for a node count."""
    for spec in _ELEM_SPECS_BY_NODE_COUNT.get(nodes_per_elem, ()):
        if spec.is_surf and spec.nodes_per_elem == nodes_per_elem:
            return spec

    raise NotImplementedError(
        f"Surface metadata is not implemented for "
        f"{nodes_per_elem}-node elements."
    )


@cache
def _get_vol_spec(nodes_per_elem: int) -> ElementSpec:
    """Return volume-element metadata for a node count."""
    for spec in _ELEM_SPECS_BY_NODE_COUNT.get(nodes_per_elem, ()):
        if not spec.is_surf and spec.nodes_per_elem == nodes_per_elem:
            return spec

    raise NotImplementedError(
        f"Volume metadata is not implemented for "
        f"{nodes_per_elem}-node elements."
    )


@cache
def _get_elem_type_from_spec(spec: ElementSpec) -> EElementType:
    """Return the registered element type for metadata."""
    for elem_type, registered_spec in ELEMENT_SPECS.items():
        if registered_spec is spec:
            return elem_type

    raise ValueError("Element specification is not registered.")


@cache
def _get_corner_idxs(nodes_per_elem: int) -> np.ndarray:
    """Return corner slots for a supported element node count."""
    _validate_nodes_per_elem(nodes_per_elem)
    if nodes_per_elem in _SURF_NODE_COUNTS:
        return np.asarray(
            _get_surf_spec(nodes_per_elem).corner_idxs,
            dtype=np.int64,
        )

    return np.asarray(
        _get_vol_spec(nodes_per_elem).corner_idxs,
        dtype=np.int64,
    )


@cache
def _get_vol_corner_idxs(nodes_per_elem: int) -> np.ndarray:
    """Return corner slots for a supported volume element."""
    return np.asarray(
        _get_vol_spec(nodes_per_elem).corner_idxs,
        dtype=np.int64,
    )


@cache
def _get_surf_map(nodes_per_elem: int) -> np.ndarray:
    """Return the local face-slot map for a volume element."""
    spec = _get_vol_spec(nodes_per_elem)
    if spec.surf_faces is None:
        raise NotImplementedError(
            f"Surface extraction is not implemented for "
            f"{spec.nodes_per_elem}-node elements."
        )

    return np.asarray(spec.surf_faces, dtype=np.int64)


@cache
def _get_elem_symmetries(
    elem_type: EElementType,
) -> tuple[tuple[int, ...], ...]:
    """Return lazily calculated proper symmetries for an element type."""
    return elem_type.calc_orient_preserving_perms()


@cache
def _get_elem_symmetry_arrs(
    elem_type: EElementType,
) -> tuple[np.ndarray, ...]:
    """Return cached array forms of an element's proper symmetries."""
    symmetry_arrs: list[np.ndarray] = []
    for perm in _get_elem_symmetries(elem_type):
        symmetry_arrs.append(np.asarray(perm, dtype=np.int64))
    return tuple(symmetry_arrs)


def _check_or_enforce_connect_table(
    connect_raw: np.ndarray,
    name: str,
    mesh_in: SimData,
    shift_all: bool,
    src_convention: MeshConvention | None,
    enforce: bool,
) -> tuple[np.ndarray, list[MeshCheckCode]]:
    """Check one table and optionally return its std representation."""
    if mesh_in.coords is None:
        raise ValueError("Mesh convention processing requires coordinates.")

    connect = _enforce_connect_arr_format(connect_raw, name)
    failures: list[MeshCheckCode] = []
    surf_only = _check_surf_mesh_type(mesh_in.mesh_type)

    if _check_transpose_needed(connect, name, mesh_in):
        failures.append(MeshCheckCode.ROW_MAJOR_CONNECTIVITY)
        connect = connect.T

    shift_needed = _check_table_needs_zero_based_shift(
        connect,
        mesh_in.coords.shape[0],
        shift_all,
    )
    if shift_needed:
        failures.append(MeshCheckCode.ZERO_BASED_INDEXING)
        connect = connect - 1

    if not _check_idxs_zero_based(connect, mesh_in.coords.shape[0]):
        failures.append(MeshCheckCode.CONNECTIVITY_INDICES)
        return connect, failures

    ordered = _enforce_node_order_table(
        connect,
        mesh_in.coords,
        surf_only=surf_only,
        src_convention=src_convention,
    )

    if not np.array_equal(ordered, connect):
        failures.append(MeshCheckCode.NODE_ORDER)

    connect = ordered

    table_is_surf = _check_surf_connect_table(
        connect,
        mesh_in.coords,
        surf_only=surf_only,
    )

    if table_is_surf:
        try:
            flips = _calc_surf_orient_flips(connect, mesh_in.coords)
        except ValueError:
            failures.append(MeshCheckCode.SURFACE_TOPOLOGY)
            if enforce:
                connect = _enforce_surf_orient_table(
                    connect,
                    mesh_in.coords,
                )
        else:
            if np.any(flips):
                failures.extend((
                    MeshCheckCode.CCW_WINDING,
                    MeshCheckCode.RIGHT_HANDED_GEOMETRY,
                ))
                if enforce:
                    connect = _apply_surf_flips(connect, flips)
    else:
        ccw_winding = _check_ccw_winding_table(
            connect,
            mesh_in.coords,
            surf_only=surf_only,
        )

        if not ccw_winding:
            failures.append(MeshCheckCode.CCW_WINDING)
            if enforce:
                connect = _enforce_ccw_winding_table(
                    connect,
                    mesh_in.coords,
                    surf_only=surf_only,
                )

        right_handed = _check_right_handed_table(
            connect,
            mesh_in.coords,
            surf_only=surf_only,
        )

        if not right_handed:
            failures.append(MeshCheckCode.RIGHT_HANDED_GEOMETRY)
            if enforce:
                connect = _enforce_right_handed_table(
                    connect,
                    mesh_in.coords,
                    surf_only=surf_only,
                )

    return np.ascontiguousarray(connect, dtype=np.int64), failures


def check_mesh_convention(
    mesh_in: SimData,
    src_convention: MeshConvention | None = None,
) -> MeshConvCheck:
    """Return failed checks for each non-conforming connectivity table."""

    if mesh_in.connect is None:
        return {}

    if mesh_in.coords is None:
        raise ValueError(
            "Mesh convention checks require 'coords' to be set.",
        )

    per_table: MeshConvCheck = {}
    shift_all = _check_mesh_needs_zero_based_shift(
        mesh_in,
        mesh_in.coords.shape[0],
    )

    for name, connect_raw in mesh_in.connect.items():
        _, failures = _check_or_enforce_connect_table(
            connect_raw,
            name,
            mesh_in,
            shift_all,
            src_convention,
            enforce=False,
        )
        if failures:
            per_table[name] = failures

    return per_table


def enforce_mesh_convention(
    mesh_in: SimData,
    src_convention: MeshConvention | None = None,
) -> SimData:
    """Return a mesh normalized to Riley's convention."""

    if mesh_in.connect is None:
        return mesh_in

    if mesh_in.coords is None:
        raise ValueError("Mesh convention enforcement requires coordinates.")

    shift_all = _check_mesh_needs_zero_based_shift(
        mesh_in,
        mesh_in.coords.shape[0],
    )

    connect_out: dict[str, np.ndarray] = {}
    changed = False
    for name, connect_raw in mesh_in.connect.items():
        connect, failure_list = _check_or_enforce_connect_table(
            connect_raw,
            name,
            mesh_in,
            shift_all,
            src_convention,
            enforce=True,
        )

        failures = frozenset(failure_list)
        if MeshCheckCode.CONNECTIVITY_INDICES in failures:
            raise ValueError(
                "Connectivity table "
                f"'{name}' contains indices outside the coordinate array after "
                "0-based normalization."
            )

        changed = changed or bool(failures)
        connect_out[name] = connect

    if not changed:
        return mesh_in

    return _copy_sim_data(mesh_in, connect=connect_out)


def _prepare_extraction_connect(
    mesh_in: SimData,
) -> tuple[dict[str, np.ndarray], bool, bool]:
    """Normalize connectivity and report the source table style."""
    if mesh_in.connect is None or mesh_in.coords is None:
        raise ValueError("Surface extraction requires a complete mesh.")

    shift_all = _check_mesh_needs_zero_based_shift(
        mesh_in,
        mesh_in.coords.shape[0],
    )
    connect_out: dict[str, np.ndarray] = {}
    src_zero_based = True
    src_row_major = True
    for name, connect_raw in mesh_in.connect.items():
        connect, failures = _check_or_enforce_connect_table(
            connect_raw,
            name,
            mesh_in,
            shift_all,
            src_convention=None,
            enforce=True,
        )

        if MeshCheckCode.CONNECTIVITY_INDICES in failures:
            raise ValueError(
                f"Connectivity table '{name}' contains invalid indices "
                "for surface extraction."
            )

        src_zero_based = src_zero_based and (
            MeshCheckCode.ZERO_BASED_INDEXING not in failures
        )

        src_row_major = src_row_major and (
            MeshCheckCode.ROW_MAJOR_CONNECTIVITY not in failures
        )

        connect_out[name] = connect

    return connect_out, src_zero_based, src_row_major


def extract_surf_mesh(
    mesh_in: SimData,
    enforce_convention: bool = True,
) -> SimData:
    """Extracts the external surface mesh from supported 3D volume elements."""

    if _check_mesh_2d(mesh_in):
        raise ValueError(
            "Surface extraction is only supported for 3D meshes. "
            "The provided mesh appears to be 2D."
        )

    if mesh_in.connect is None:
        raise ValueError("Surface extraction requires connectivity tables.")

    if mesh_in.coords is None:
        raise ValueError("Surface extraction requires coordinates.")

    connect_norm, src_zero_based, src_row_major = (
        _prepare_extraction_connect(mesh_in)
    )

    surf_connect_glob: dict[str, np.ndarray] = {}
    surf_elem_srcs: dict[str, np.ndarray] = {}
    surf_node_blocks: list[np.ndarray] = []

    for name, connect in connect_norm.items():
        (surf_faces, surf_parent_elem_idxs) = _extract_surf_faces_from_table(
            connect,
            mesh_in.coords,
        )

        surf_connect_glob[name] = surf_faces
        surf_elem_srcs[name] = surf_parent_elem_idxs

        if surf_faces.size:
            surf_node_blocks.append(surf_faces.reshape(-1))

    if surf_node_blocks:
        surf_node_idxs = np.unique(np.concatenate(surf_node_blocks))
    else:
        surf_node_idxs = np.array([], dtype=np.int64)

    surf_coords = np.ascontiguousarray(
        mesh_in.coords[surf_node_idxs],
        dtype=mesh_in.coords.dtype,
    )

    coord_remap = np.full(mesh_in.coords.shape[0], -1, dtype=np.int64)
    coord_remap[surf_node_idxs] = np.arange(
        surf_node_idxs.shape[0],
        dtype=np.int64,
    )

    surf_connect_loc: dict[str, np.ndarray] = {}
    for name, surf_faces in surf_connect_glob.items():
        surf_connect_loc[name] = coord_remap[surf_faces]

    surf_mesh = _copy_sim_data(mesh_in)
    surf_mesh.coords = surf_coords
    surf_mesh.connect = surf_connect_loc
    surf_mesh.side_sets = None
    surf_mesh.update_mesh_type()

    if mesh_in.node_vars is not None:
        surf_mesh.node_vars = {}
        for name, values in mesh_in.node_vars.items():
            surf_mesh.node_vars[name] = values[surf_node_idxs, :]

    if mesh_in.elem_vars is not None:
        surf_mesh.elem_vars = {}
        for (name, block_id), values in mesh_in.elem_vars.items():
            connect_key = f"connect{block_id}"
            if connect_key in surf_elem_srcs:
                surf_mesh.elem_vars[(name, block_id)] = values[
                    surf_elem_srcs[connect_key], :
                ]

    if not enforce_convention:
        surf_mesh = _restore_src_connect_style(
            surf_mesh,
            one_based=not src_zero_based,
            transposed=not src_row_major,
        )

        return surf_mesh

    return surf_mesh


def extract_surf_between(
    mesh_in: SimData,
    point: np.ndarray | list[float] | tuple[float, ...],
    normal: np.ndarray | list[float] | tuple[float, ...],
    distance: float | None = None,
    tolerance: float = 1.0e-6,
    enforce_convention: bool = True,
) -> SimData:
    """Extract a surface mesh between two parallel planes."""

    if mesh_in.connect is None:
        raise ValueError("Surface extraction requires connectivity tables.")

    if mesh_in.coords is None:
        raise ValueError("Surface extraction requires coordinates.")

    point_arr = _conv_to_vec3(point, "point")
    normal_arr = _conv_to_vec3(normal, "normal")
    normal_magnitude = np.linalg.norm(normal_arr)

    if normal_magnitude < _TOL.geom:
        raise ValueError("Normal vector cannot be zero.")

    normal_arr = normal_arr / normal_magnitude

    if not np.isfinite(tolerance) or tolerance < 0.0:
        raise ValueError("'tolerance' must be a finite non-negative value.")

    if distance is not None and not np.isfinite(distance):
        raise ValueError("'distance' must be finite when provided.")

    coord_offsets = mesh_in.coords - point_arr
    projs = coord_offsets @ normal_arr

    # Determine bounds
    if distance is not None:
        plane_distance = float(distance)
        min_bound = min(0.0, plane_distance) - tolerance
        max_bound = max(0.0, plane_distance) + tolerance
    else:
        min_bound = -tolerance
        max_bound = tolerance

    connect_norm, src_zero_based, src_row_major = (
        _prepare_extraction_connect(mesh_in)
    )

    surf_connect_glob: dict[str, np.ndarray] = {}
    surf_elem_srcs: dict[str, np.ndarray] = {}
    surf_node_blocks: list[np.ndarray] = []

    for name, connect in connect_norm.items():
        nodes_per_elem = connect.shape[1]
        is_vol = _check_vol_connect_table(connect, mesh_in.coords)
        if is_vol:
            face_map = _get_surf_map(nodes_per_elem)
            faces_per_elem = face_map.shape[0]
            faces_wound = connect[:, face_map]
            faces_flat_wound = faces_wound.reshape((-1, face_map.shape[1]))
            faces_flat_sorted = np.sort(faces_flat_wound, axis=1)

            _, unique_idxs = np.unique(
                faces_flat_sorted,
                axis=0,
                return_index=True,
            )
            candidate_faces = faces_flat_wound[unique_idxs]
            parent_idxs = unique_idxs // faces_per_elem

        else:
            candidate_faces = connect
            parent_idxs = np.arange(connect.shape[0], dtype=np.int64)

        if candidate_faces.size == 0:
            continue

        face_projs = projs[candidate_faces]
        in_bounds = np.all(
            (face_projs >= min_bound)
            & (face_projs <= max_bound),
            axis=1
        )

        filtered_faces = candidate_faces[in_bounds]
        filtered_parents = parent_idxs[in_bounds]

        if filtered_faces.size > 0:
            surf_connect_glob[name] = filtered_faces
            surf_elem_srcs[name] = filtered_parents
            surf_node_blocks.append(filtered_faces.reshape(-1))

    if surf_node_blocks:
        surf_node_idxs = np.unique(np.concatenate(surf_node_blocks))
    else:
        surf_node_idxs = np.array([], dtype=np.int64)

    if len(surf_connect_glob) == 0 or surf_node_idxs.size == 0:
        raise ValueError(
            "No elements/faces found between the specified planes."
        )

    surf_coords = np.ascontiguousarray(
        mesh_in.coords[surf_node_idxs],
        dtype=mesh_in.coords.dtype
    )
    coord_remap = np.full(mesh_in.coords.shape[0], -1, dtype=np.int64)
    coord_remap[surf_node_idxs] = np.arange(
        surf_node_idxs.shape[0], dtype=np.int64
    )

    surf_connect_loc: dict[str, np.ndarray] = {}
    for name, surf_faces in surf_connect_glob.items():
        surf_connect_loc[name] = coord_remap[surf_faces]

    surf_mesh = _copy_sim_data(mesh_in)
    surf_mesh.coords = surf_coords
    surf_mesh.connect = surf_connect_loc
    surf_mesh.side_sets = None
    surf_mesh.update_mesh_type()

    if mesh_in.node_vars is not None:
        surf_mesh.node_vars = {}
        for name, values in mesh_in.node_vars.items():
            surf_mesh.node_vars[name] = values[surf_node_idxs, :]

    if mesh_in.elem_vars is not None:
        surf_mesh.elem_vars = {}
        for (var_name, block_id), values in mesh_in.elem_vars.items():
            connect_key = f"connect{block_id}"
            if connect_key in surf_elem_srcs:
                surf_mesh.elem_vars[(var_name, block_id)] = values[
                    surf_elem_srcs[connect_key], :
                ]

    if not enforce_convention:
        surf_mesh = _restore_src_connect_style(
            surf_mesh,
            one_based=not src_zero_based,
            transposed=not src_row_major,
        )
        return surf_mesh

    return enforce_mesh_convention(surf_mesh)
