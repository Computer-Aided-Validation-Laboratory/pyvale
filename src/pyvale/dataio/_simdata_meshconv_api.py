# --------------------------------------------------------------------------
# Riley: A High Performance Rasteriser for DIC UQ
#
# Copyright (c) 2025-2026 scepticalrabbit (Lloyd Fletcher)
# Licensed under the MIT License (see LICENSE file for details)
#
# Authors: scepticalrabbit (Lloyd Fletcher)
# --------------------------------------------------------------------------
"""Internal implementation retained for the existing ``SimData`` contract.

This module is private to Pyvale DataIO. New render code uses Riley's public
mesh conversion API directly.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from types import MappingProxyType
from typing import Literal

import numpy as np

from pyvale.dataio import _simdata_meshconv_impl as _meshconv
from pyvale.dataio._simdata_meshconv_impl import (
    EElementType,
    EMeshType,
    MeshCheckCode,
    MeshConvCheck,
    MeshConvention,
    MeshConvErr,
    SimData,
)


class EConnectAxis(Enum):
    """Axis containing elements in a source connectivity array."""

    ROW = "row"
    COLUMN = "column"


class ENodeOrder(Enum):
    """Built-in local-node ordering adapters."""

    RILEY = "riley"
    VTK = "vtk"
    EXODUS = "exodus"


@dataclass(frozen=True, slots=True)
class EdgeNode:
    """Identify a source edge-node slot by its source corner slots."""

    corners: tuple[int, int]
    node_slot: int


@dataclass(frozen=True, slots=True)
class FaceNode:
    """Identify a source face-node slot by its source corner slots."""

    corners: tuple[int, ...]
    node_slot: int


@dataclass(frozen=True, slots=True)
class UserTopology:
    """Describe local topology using slots in a source connectivity row.

    ``corner_slots`` must follow a valid source topological ordering. Edge and
    face nodes are associated with the source corner slots bounding their
    topological entity. No Riley slot numbers are required.
    """

    corner_slots: tuple[int, ...]
    edge_nodes: tuple[EdgeNode, ...] = ()
    face_nodes: tuple[FaceNode, ...] = ()
    centre_slot: int | None = None


@dataclass(frozen=True, slots=True)
class ConnectConvention:
    """Describe how to interpret one source connectivity array.

    ``elem_axis`` says whether each element occupies a row or a column, and
    ``index_base`` must explicitly be zero or one. ``node_order`` selects a
    named source adapter or a ``UserTopology``; Riley does not guess it.

    For an open surface, source winding declares the material-facing side.
    Set ``reverse_open_surface`` to choose its opposite, or provide a
    non-zero ``material_normal_hint`` to choose the side whose component
    normal has a positive dot product with that vector. These options do not
    alter the exterior/cavity orientation of closed shells.

    One convention describes one connectivity table. Reuse the same instance
    for multiple tables with the same representation, or provide a mapping of
    table names to conventions in calling code when their representations
    differ.
    """

    elem_type: EElementType
    elem_axis: EConnectAxis
    index_base: Literal[0, 1]
    node_order: ENodeOrder | UserTopology = ENodeOrder.RILEY
    material_normal_hint: tuple[float, float, float] | None = None
    reverse_open_surface: bool = False

    def __post_init__(self) -> None:
        """Verify convention fields at construction time."""
        if not isinstance(self.elem_type, EElementType):
            raise TypeError("elem_type must be an EElementType member.")
        if not isinstance(self.elem_axis, EConnectAxis):
            raise TypeError("elem_axis must be an EConnectAxis member.")
        if isinstance(self.index_base, bool) or self.index_base not in (0, 1):
            raise ValueError("index_base must be the integer 0 or 1.")
        valid_order = isinstance(self.node_order, (ENodeOrder, UserTopology))
        if not valid_order:
            raise TypeError("node_order must be ENodeOrder or UserTopology.")


@dataclass(frozen=True, slots=True)
class MeshGeometry:
    """A single-topology mesh using Riley's standard convention."""

    elem_type: EElementType
    coords: np.ndarray
    connect: np.ndarray


@dataclass(frozen=True, slots=True)
class MeshVerifyIssue:
    """One independently detected Riley mesh verification failure."""

    code: str
    message: str
    elem_idx: int | None = None


class MeshVerifyErr(ValueError):
    """Collect all mesh verification failures found in one pass."""

    def __init__(self, issues: list[MeshVerifyIssue]) -> None:
        self.issues = tuple(issues)
        lines = [f"Mesh verification failed with {len(issues)} issue(s):"]
        for issue in issues:
            location = ""
            if issue.elem_idx is not None:
                location = f" element {issue.elem_idx}:"
            lines.append(f"- [{issue.code}]{location} {issue.message}")
        super().__init__("\n".join(lines))


_VTK_TO_RILEY = MappingProxyType({
    elem_type: tuple(range(elem_type.calc_ref_coords().shape[0]))
    for elem_type in EElementType
})

# Exodus uses bottom, vertical, then top HEX edge groups. Riley and VTK use
# bottom, top, then vertical edge groups. Other supported Exodus topology
# mappings are identical to Riley's standard local roles.
_EXODUS_TO_RILEY = MappingProxyType({
    **_VTK_TO_RILEY,
    EElementType.HEX20: (
        0, 1, 2, 3, 4, 5, 6, 7,
        8, 9, 10, 11, 16, 17, 18, 19, 12, 13, 14, 15,
    ),
})


def check_mesh_convention(
    mesh_in: SimData,
    src_convention: MeshConvention | None = None,
) -> MeshConvCheck:
    """Check a mesh for conformance to Riley's mesh convention.

    This function inspects all connectivity tables in the mesh and reports
    which convention checks fail. An empty dictionary means the mesh fully
    conforms to Riley's convention.

    Parameters
    ----------
    mesh_in : SimData
        The mesh to check. Must have ``coords`` (N x 3 array) and ``connect``
        (dict of connectivity tables). Each connectivity table must be
        2D with one element per row and one local node per column.
    src_convention : MeshConvention | None, optional
        Source ordering for specific element types. Each permutation satisfies
        ``riley_row[target_slot] = source_row[permutation[target_slot]]``.
        Omitted element types are assumed to use Riley ordering. If None, all
        connectivity is assumed to use Riley ordering.

    Returns
    -------
    MeshConvCheck
        Dictionary mapping connectivity table names to lists of failed
        ``MeshCheckCode`` conditions. Empty dict means the mesh conforms.
        Possible failure codes:

        - ``ROW_MAJOR_CONNECTIVITY``: table is column-major, needs transpose
        - ``ZERO_BASED_INDEXING``: indices are 1-based, need 0-based shift
        - ``CONNECTIVITY_INDICES``: indices outside valid range [0, N-1]
        - ``CCW_WINDING``: surface elements not wound counter-clockwise
        - ``RIGHT_HANDED_GEOMETRY``: volume elements not right-handed
        - ``SURFACE_TOPOLOGY``: non-manifold or unorientable surface
        - ``NODE_ORDER``: higher-order nodes in wrong slots for element type

    Examples
    --------
    >>> from riley.python import meshconv
    >>> import numpy as np
    >>> coords = np.array(
    ...     [[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0]], dtype=np.float64
    ... )
    >>> mesh = meshconv.SimData(
    ...     coords=coords, connect={"connect1": np.array([[0, 1, 2, 3]])}
    ... )
    >>> report = meshconv.check_mesh_convention(mesh)
    >>> report == {}
    True
    """
    return _meshconv.check_mesh_convention(mesh_in, src_convention)


def enforce_mesh_convention(
    mesh_in: SimData,
    src_convention: MeshConvention | None = None,
) -> SimData:
    """Normalise a mesh to Riley's mesh convention.

    Applies all necessary corrections to bring a mesh into conformance with
    Riley's convention. Only fixes the conditions reported by
    :func:`check_mesh_convention`, and applies them in a fixed order
    so the result is deterministic.

    Parameters
    ----------
    mesh_in : SimData
        The mesh to normalise. Must have ``coords`` and ``connect`` set.
    src_convention : MeshConvention | None, optional
        Source ordering for specific element types. Each permutation satisfies
        ``riley_row[target_slot] = source_row[permutation[target_slot]]``.
        Omitted element types are assumed to use Riley ordering. If None, all
        connectivity is assumed to use Riley ordering.

    Returns
    -------
    SimData
        A new ``SimData`` instance with normalised connectivity tables, or the
        original mesh instance if it already conformed (idempotent).

    Notes
    -----
    The normalisation enforces:

    - **Row-major connectivity**: each element is one row,
      local nodes are columns
    - **Zero-based indexing**: all indices in range [0, N-1] where N = num nodes
    - **Node order**: higher-order nodes (mid-edge, mid-face, centre) placed in
      Riley's standard slots per element type
    - **Surface winding**: CCW when viewed from material-facing side (outward
      for closed shells, inward for cavity boundaries)
    - **Volume handedness**: right-handed coordinate system for TET/HEX elements
    - **Index validity**: all indices reference existing coordinate rows

    If ``CONNECTIVITY_INDICES`` would be violated (index out of bounds after
    normalization), a ``ValueError`` is raised instead of silently corrupting
    the mesh.

    Examples
    --------
    >>> from riley.python import meshconv
    >>> import numpy as np
    >>> coords = np.array(
    ...     [[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0]], dtype=np.float64
    ... )
    >>> # 1-based indexing, needs conversion
    >>> mesh = meshconv.SimData(
    ...     coords=coords, connect={"connect1": np.array([[1, 2, 3, 4]])}
    ... )
    >>> mesh_out = meshconv.enforce_mesh_convention(mesh)
    >>> mesh_out.connect["connect1"]
    array([[0, 1, 2, 3]])
    >>> meshconv.check_mesh_convention(mesh_out) == {}
    True
    """
    return _meshconv.enforce_mesh_convention(mesh_in, src_convention)


def infer_mesh_convention(mesh_in: SimData) -> MeshConvention:
    """Infer a source convention from affine element geometry.

    Inference is intentionally conservative and rejects ambiguous or mixed
    source layouts. Prefer an explicitly declared :class:`MeshConvention`
    whenever the source format is known.

    Parameters
    ----------
    mesh_in : SimData
        Mesh whose source connectivity order should be inferred.

    Returns
    -------
    MeshConvention
        Inferred source-slot mapping for each element family in the mesh.

    Raises
    ------
    MeshConvErr
        If the source node roles cannot be inferred unambiguously.
    """
    return _meshconv.infer_mesh_convention(mesh_in)


def extract_surf_mesh(
    mesh_in: SimData,
    enforce_convention: bool = True,
) -> SimData:
    """Extract the external surface mesh from a 3D volume mesh.

    For volume element types (TET4, TET10, HEX8, HEX20, HEX27), this function
    identifies boundary faces (those belonging to only one element) and returns
    them as a surface mesh with proper outward-facing normals.

    Parameters
    ----------
    mesh_in : SimData
        The input 3D volume mesh. Must have ``coords`` and ``connect`` with
        volume element types.
    enforce_convention : bool, optional
        If True (default), the output surface mesh is normalised to Riley's
        convention (0-based, CCW winding, proper node order). If False, the
        output retains the input mesh's indexing style (1-based vs 0-based,
        row-major vs column-major).

    Returns
    -------
    SimData
        A surface mesh containing only the boundary faces. The returned mesh
        has:

        - ``coords``: subset of input coordinates used by boundary faces
        - ``connect``: boundary face connectivity (QUAD4/8/9 or TRI3/6/7)
        - ``mesh_type``: ``EMeshType.SURF``
        - ``node_vars``: interpolated nodal variables for surface nodes
        - ``elem_vars``: element variables for boundary faces

    Raises
    ------
    ValueError
        If the input mesh is 2D (surface elements only) or has no connectivity.

    Examples
    --------
    >>> from riley.python import meshconv
    >>> import numpy as np
    >>> coords = np.array((
    ...     (0.0, 0.0, 0.0), (1.0, 0.0, 0.0),
    ...     (1.0, 1.0, 0.0), (0.0, 1.0, 0.0),
    ...     (0.0, 0.0, 1.0), (1.0, 0.0, 1.0),
    ...     (1.0, 1.0, 1.0), (0.0, 1.0, 1.0),
    ... ))
    >>> connect = np.arange(8, dtype=np.int64).reshape(1, 8)
    >>> mesh = meshconv.SimData(coords=coords, connect={"connect1": connect})
    >>> surf = meshconv.extract_surf_mesh(mesh)
    >>> surf.mesh_type is meshconv.EMeshType.SURF
    True
    >>> list(surf.connect.values())[0].shape  # (faces, nodes per face)
    (6, 4)
    """
    return _meshconv.extract_surf_mesh(mesh_in, enforce_convention)


def extract_surf_between(
    mesh_in: SimData,
    point: np.ndarray | list[float] | tuple[float, ...],
    normal: np.ndarray | list[float] | tuple[float, ...],
    distance: float | None = None,
    tolerance: float = 1.0e-6,
    enforce_convention: bool = True,
) -> SimData:
    """Extract a surface mesh from a slice between two parallel planes.

    For volume meshes, extracts internal faces between elements where nodes
    lie between the two planes. For surface meshes, extracts faces whose
    nodes lie between the planes. The output is a surface mesh with proper
    outward-facing normals.

    Parameters
    ----------
    mesh_in : SimData
        The input simulation data/mesh. Can be volume or surface elements.
    point : np.ndarray | list[float] | tuple[float, ...]
        A point on the first plane. Only the first 3 components are used.
    normal : np.ndarray | list[float] | tuple[float, ...]
        The normal vector defining the plane orientation. Must be non-zero.
        Only the first 3 components are used.
    distance : float | None, optional
        Distance along the normal to the second plane. If None (default),
        extracts a thin slice at the first plane with ``+/- tolerance``
        thickness. If provided, extracts the slab between the two planes.
    tolerance : float, optional
        Numerical tolerance for checking if nodes lie between the planes.
        Defaults to 1.0e-6. Nodes with projected distance in
        ``[min_bound - tol, max_bound + tol]`` are included.
    enforce_convention : bool, optional
        If True (default), normalises the output mesh to Riley's convention.
        If False, retains the input mesh's indexing style.

    Returns
    -------
    SimData
        The extracted surface mesh containing faces/elements between the
        planes. The mesh has ``mesh_type = EMeshType.SURF`` and conforming
        connectivity (if ``enforce_convention=True``).

    Raises
    ------
    ValueError
        If no elements/faces are found between the planes, if normal is zero,
        or if mesh lacks required coordinates/connectivity.

    Examples
    --------
    >>> from riley.python import meshconv
    >>> import numpy as np
    >>> coords = np.array((
    ...     (0.0, 0.0, 0.0), (1.0, 0.0, 0.0),
    ...     (1.0, 1.0, 0.0), (0.0, 1.0, 0.0),
    ...     (0.0, 0.0, 1.0), (1.0, 0.0, 1.0),
    ...     (1.0, 1.0, 1.0), (0.0, 1.0, 1.0),
    ... ))
    >>> connect = np.arange(8, dtype=np.int64).reshape(1, 8)
    >>> mesh = meshconv.SimData(coords=coords, connect={"connect1": connect})
    >>> slab = meshconv.extract_surf_between(
    ...     mesh, point=(0.0, 0.0, 0.0), normal=(0.0, 0.0, 1.0)
    ... )
    >>> slab.connect["connect1"].shape
    (1, 4)
    """
    return _meshconv.extract_surf_between(
        mesh_in,
        point,
        normal,
        distance=distance,
        tolerance=tolerance,
        enforce_convention=enforce_convention,
    )


def _get_user_topology_perm(
    elem_type: EElementType,
    topology: UserTopology,
) -> tuple[int, ...]:
    """Build a Riley-target-to-source permutation from source topology."""
    spec = _meshconv.ELEMENT_SPECS[elem_type]
    node_count = spec.nodes_per_elem
    corner_count = len(spec.corner_idxs)
    if len(topology.corner_slots) != corner_count:
        raise MeshConvErr(
            f"{elem_type.value} requires {corner_count} corner slots."
        )

    perm = [-1] * node_count
    for target_slot, source_slot in zip(
        spec.corner_idxs,
        topology.corner_slots,
        strict=True,
    ):
        perm[target_slot] = source_slot

    edge_lookup: dict[frozenset[int], int] = {}
    for edge in topology.edge_nodes:
        edge_lookup[frozenset(edge.corners)] = edge.node_slot

    edge_pairs = spec.edge_pairs
    if not edge_pairs and node_count > corner_count:
        edge_pairs = tuple(
            (idx, (idx + 1) % corner_count)
            for idx in range(corner_count)
        )

    for target_slot, (start, end) in enumerate(
        edge_pairs,
        start=corner_count,
    ):
        source_edge = frozenset((
            topology.corner_slots[start],
            topology.corner_slots[end],
        ))
        if source_edge not in edge_lookup:
            raise MeshConvErr(
                f"{elem_type.value} topology is missing edge "
                f"{tuple(sorted(source_edge))}."
            )
        perm[target_slot] = edge_lookup[source_edge]

    face_lookup: dict[frozenset[int], int] = {}
    for face in topology.face_nodes:
        face_lookup[frozenset(face.corners)] = face.node_slot

    for target_slot, face_corners in zip(
        spec.face_centre_idxs,
        spec.face_corner_idxs,
        strict=True,
    ):
        source_face = frozenset(
            topology.corner_slots[idx] for idx in face_corners
        )
        if source_face not in face_lookup:
            raise MeshConvErr(
                f"{elem_type.value} topology is missing face "
                f"{tuple(sorted(source_face))}."
            )
        perm[target_slot] = face_lookup[source_face]

    centre_slot = spec.cell_centre_idx
    if centre_slot is None:
        centre_slot = spec.centre_idx
    if centre_slot is not None:
        if topology.centre_slot is None:
            raise MeshConvErr(
                f"{elem_type.value} topology requires a centre slot."
            )
        perm[centre_slot] = topology.centre_slot

    if set(perm) != set(range(node_count)):
        raise MeshConvErr(
            f"{elem_type.value} topology must use every source slot exactly "
            "once."
        )
    return tuple(perm)


def _get_source_perm(convention: ConnectConvention) -> tuple[int, ...]:
    """Return the Riley-target-to-source permutation for a convention."""
    if isinstance(convention.node_order, UserTopology):
        return _get_user_topology_perm(
            convention.elem_type,
            convention.node_order,
        )
    if convention.node_order in (ENodeOrder.RILEY, ENodeOrder.VTK):
        return _VTK_TO_RILEY[convention.elem_type]
    if convention.node_order is ENodeOrder.EXODUS:
        if convention.elem_type is EElementType.HEX27:
            raise MeshConvErr(
                "The Exodus HEX27 adapter requires an authoritative fixture; "
                "supply UserTopology explicitly."
            )
        return _EXODUS_TO_RILEY[convention.elem_type]
    raise MeshConvErr(f"Unsupported node ordering: {convention.node_order}.")


def _enforce_surface_orient(
    connect: np.ndarray,
    coords: np.ndarray,
) -> np.ndarray:
    """Make faces coherent while retaining each open component's side."""
    topology = _meshconv._build_surf_topology(
        connect,
        np.arange(connect.shape[0], dtype=np.int64),
    )
    closed_flips = _meshconv._calc_surf_orient_flips(connect, coords)
    flips = np.zeros(connect.shape[0], dtype=bool)
    for rows, edge_keys in topology:
        is_closed = all(len(edge_keys[key]) == 2 for key in edge_keys)
        if is_closed:
            flips[rows] = closed_flips[rows]
        else:
            flips[rows] = _meshconv._calc_comp_rel_flips(rows, edge_keys)
    return _meshconv._apply_surf_flips(connect, flips)


def _orient_surface_to_hint(
    connect: np.ndarray,
    coords: np.ndarray,
    hint: np.ndarray,
) -> np.ndarray:
    """Orient each connected surface component towards a normal hint."""
    topology = _meshconv._build_surf_topology(
        connect,
        np.arange(connect.shape[0], dtype=np.int64),
    )
    flips = np.zeros(connect.shape[0], dtype=bool)
    for rows, edge_keys in topology:
        is_closed = all(len(edge_keys[key]) == 2 for key in edge_keys)
        if is_closed:
            continue
        first_coords = coords[connect[rows[0]]]
        normal = _meshconv._calc_face_normal(first_coords)
        if np.dot(normal, hint) < 0.0:
            flips[rows] = True
    return _meshconv._apply_surf_flips(connect, flips)


def _reverse_open_surface(
    connect: np.ndarray,
) -> np.ndarray:
    """Reverse open components without changing closed-shell orientation."""
    topology = _meshconv._build_surf_topology(
        connect,
        np.arange(connect.shape[0], dtype=np.int64),
    )
    flips = np.zeros(connect.shape[0], dtype=bool)
    for rows, edge_keys in topology:
        is_closed = all(len(edge_keys[key]) == 2 for key in edge_keys)
        if not is_closed:
            flips[rows] = True
    return _meshconv._apply_surf_flips(connect, flips)


def convert_mesh(
    coords: np.ndarray,
    connect: np.ndarray,
    convention: ConnectConvention,
) -> MeshGeometry:
    """Convert NumPy mesh arrays to Riley's standard convention.

    Parameters
    ----------
    coords : np.ndarray
        Source coordinates with shape ``(nodes, 3)``.
    connect : np.ndarray
        One connectivity table containing exactly one element topology.
    convention : ConnectConvention
        Explicit source element type, element axis, index base and local-node
        ordering. A built-in adapter or explicit ``UserTopology`` is required.

    Returns
    -------
    MeshGeometry
        A new mesh in Riley's standard convention. Coordinate rows and global
        node IDs are not compacted or otherwise reordered.
    """
    coords_in = np.asarray(coords)
    connect_in = np.asarray(connect)
    if coords_in.ndim != 2 or coords_in.shape[1] != 3:
        raise MeshConvErr("coords must have shape (nodes, 3).")
    if coords_in.shape[0] == 0:
        raise MeshConvErr("coords must contain at least one node.")
    if not np.issubdtype(coords_in.dtype, np.floating):
        raise MeshConvErr("coords must have a floating-point dtype.")
    if not np.all(np.isfinite(coords_in)):
        raise MeshConvErr("coords must contain only finite values.")
    if connect_in.ndim != 2:
        raise MeshConvErr("connect must be a two-dimensional array.")
    if connect_in.size == 0 or connect_in.shape[0] == 0:
        raise MeshConvErr("connect must contain at least one element.")
    if np.issubdtype(connect_in.dtype, np.bool_) or not np.issubdtype(
        connect_in.dtype,
        np.integer,
    ):
        raise MeshConvErr("connect must have an integer dtype.")

    if convention.elem_axis is EConnectAxis.COLUMN:
        connect_in = connect_in.T
    spec = _meshconv.ELEMENT_SPECS[convention.elem_type]
    if connect_in.shape[1] != spec.nodes_per_elem:
        raise MeshConvErr(
            f"{convention.elem_type.value} connectivity requires "
            f"{spec.nodes_per_elem} nodes per element."
        )

    if np.issubdtype(connect_in.dtype, np.unsignedinteger):
        max_int64 = np.iinfo(np.int64).max
        if connect_in.size and int(np.max(connect_in)) > max_int64:
            raise MeshConvErr("connect contains an index outside int64 range.")
    connect_std = np.ascontiguousarray(connect_in, dtype=np.int64)
    connect_std = connect_std - convention.index_base
    if np.any(connect_std < 0) or np.any(connect_std >= coords_in.shape[0]):
        raise MeshConvErr(
            "connect contains an index outside the coordinate array."
        )

    source_perm = np.asarray(_get_source_perm(convention), dtype=np.int64)
    connect_std = np.ascontiguousarray(
        connect_std[:, source_perm],
        dtype=np.int64,
    )
    mesh_type = EMeshType.SURF if spec.is_surf else EMeshType.VOL
    mesh_legacy = SimData(
        coords=np.ascontiguousarray(coords_in, dtype=np.float64),
        connect={"connect": connect_std},
        mesh_type=mesh_type,
    )
    if spec.is_surf:
        connect_std = _enforce_surface_orient(
            connect_std,
            mesh_legacy.coords,
        )
        mesh_legacy.connect["connect"] = connect_std
    else:
        mesh_legacy = _meshconv.enforce_mesh_convention(mesh_legacy)
    if mesh_legacy.connect is None:
        raise MeshConvErr("Mesh conversion did not produce connectivity.")
    connect_std = mesh_legacy.connect["connect"]

    if spec.is_surf and convention.reverse_open_surface:
        connect_std = _reverse_open_surface(connect_std)
    if spec.is_surf and convention.material_normal_hint is not None:
        hint = np.asarray(convention.material_normal_hint, dtype=np.float64)
        if hint.shape != (3,) or not np.all(np.isfinite(hint)):
            raise MeshConvErr("material_normal_hint must be a finite vector3.")
        if np.linalg.norm(hint) <= 1.0e-12:
            raise MeshConvErr("material_normal_hint must be non-zero.")
        connect_std = _orient_surface_to_hint(
            connect_std,
            mesh_legacy.coords,
            hint,
        )

    mesh_out = MeshGeometry(
        elem_type=convention.elem_type,
        coords=mesh_legacy.coords,
        connect=np.ascontiguousarray(connect_std, dtype=np.int64),
    )
    verify_mesh(mesh_out)
    return mesh_out


def verify_mesh(mesh: MeshGeometry) -> None:
    """Verify Riley's standard mesh convention and report all found issues.

    Independent structural and per-element failures are collected into one
    ``MeshVerifyErr`` so a user can correct several input problems at once.
    Checks that depend on unsafe or malformed arrays are skipped.
    """
    issues: list[MeshVerifyIssue] = []
    if not isinstance(mesh.elem_type, EElementType):
        issues.append(MeshVerifyIssue(
            "element_type", "elem_type must be an EElementType member."
        ))
        raise MeshVerifyErr(issues)

    spec = _meshconv.ELEMENT_SPECS[mesh.elem_type]
    coords = np.asarray(mesh.coords)
    connect = np.asarray(mesh.connect)
    coords_safe = coords.ndim == 2 and coords.shape[1:] == (3,)
    connect_safe = connect.ndim == 2

    if not coords_safe:
        issues.append(MeshVerifyIssue(
            "coordinate_shape", "coords must have shape (nodes, 3)."
        ))
    else:
        if coords.shape[0] == 0:
            issues.append(MeshVerifyIssue(
                "empty_coordinates", "coords must contain at least one node."
            ))
        if coords.dtype != np.float64:
            issues.append(MeshVerifyIssue(
                "coordinate_dtype", "coords must have dtype float64."
            ))
        if not coords.flags.c_contiguous:
            issues.append(MeshVerifyIssue(
                "coordinate_layout", "coords must be C-contiguous."
            ))
        if not np.all(np.isfinite(coords)):
            issues.append(MeshVerifyIssue(
                "coordinate_values", "coords must contain only finite values."
            ))

    if not connect_safe:
        issues.append(MeshVerifyIssue(
            "connectivity_shape", "connect must be two-dimensional."
        ))
    else:
        if connect.shape[0] == 0:
            issues.append(MeshVerifyIssue(
                "empty_connectivity", "connect must contain an element."
            ))
        if connect.shape[1] != spec.nodes_per_elem:
            issues.append(MeshVerifyIssue(
                "connectivity_width",
                f"{mesh.elem_type.value} requires {spec.nodes_per_elem} "
                "nodes per element.",
            ))
            connect_safe = False
        if connect.dtype != np.int64:
            issues.append(MeshVerifyIssue(
                "connectivity_dtype", "connect must have dtype int64."
            ))
            connect_safe = False
        if not connect.flags.c_contiguous:
            issues.append(MeshVerifyIssue(
                "connectivity_layout", "connect must be C-contiguous."
            ))

    if coords_safe and connect_safe and connect.size:
        valid_idxs = np.logical_and(connect >= 0, connect < coords.shape[0])
        if not np.all(valid_idxs):
            issues.append(MeshVerifyIssue(
                "connectivity_indices",
                "connect contains indices outside the coordinate array.",
            ))
        else:
            for elem_idx, row in enumerate(connect):
                if np.unique(row).shape[0] != row.shape[0]:
                    issues.append(MeshVerifyIssue(
                        "duplicate_node",
                        "connectivity contains a duplicate node ID.",
                        elem_idx,
                    ))
                    continue
                if spec.is_surf:
                    try:
                        _meshconv._calc_face_normal(coords[row])
                    except ValueError as error:
                        issues.append(MeshVerifyIssue(
                            "surface_geometry", str(error), elem_idx
                        ))

            if spec.is_surf:
                try:
                    topology = _meshconv._build_surf_topology(
                        connect,
                        np.arange(connect.shape[0], dtype=np.int64),
                    )
                    expected = _enforce_surface_orient(connect, coords)
                except (ValueError, NotImplementedError) as error:
                    issues.append(MeshVerifyIssue(
                        "surface_topology",
                        str(error),
                    ))
                else:
                    if not np.array_equal(expected, connect):
                        issues.append(MeshVerifyIssue(
                            "surface_orientation",
                            "surface faces are inconsistent or a closed shell "
                            "is not material-facing.",
                        ))
                    if not topology:
                        issues.append(MeshVerifyIssue(
                            "surface_topology", "surface has no components."
                        ))
            else:
                for elem_idx, row in enumerate(connect):
                    corner_idxs = np.asarray(spec.corner_idxs, dtype=np.int64)
                    metric = _meshconv._calc_vol_signed_metric(
                        coords[row[corner_idxs]]
                    )
                    if metric <= 1.0e-12:
                        issues.append(MeshVerifyIssue(
                            "volume_orientation",
                            "volume element is degenerate or not right-handed.",
                            elem_idx,
                        ))

    if issues:
        raise MeshVerifyErr(issues)


_VOLUME_SURFACE_TYPES = MappingProxyType({
    EElementType.TET4: EElementType.TRI3,
    EElementType.TET10: EElementType.TRI6,
    EElementType.HEX8: EElementType.QUAD4,
    EElementType.HEX20: EElementType.QUAD8,
    EElementType.HEX27: EElementType.QUAD9,
})


def extract_surface(mesh: MeshGeometry) -> MeshGeometry:
    """Extract a compact Riley-standard surface from a volume mesh."""
    verify_mesh(mesh)
    if mesh.elem_type not in _VOLUME_SURFACE_TYPES:
        raise MeshConvErr("extract_surface requires a volume mesh.")
    legacy = SimData(
        coords=mesh.coords,
        connect={"connect": mesh.connect},
        mesh_type=EMeshType.VOL,
    )
    surface = _meshconv.extract_surf_mesh(legacy)
    if surface.coords is None or surface.connect is None:
        raise MeshConvErr("Surface extraction did not produce a mesh.")
    mesh_out = MeshGeometry(
        elem_type=_VOLUME_SURFACE_TYPES[mesh.elem_type],
        coords=surface.coords,
        connect=surface.connect["connect"],
    )
    verify_mesh(mesh_out)
    return mesh_out


__all__ = [
    "ConnectConvention",
    "EConnectAxis",
    "EElementType",
    "EMeshType",
    "ENodeOrder",
    "EdgeNode",
    "FaceNode",
    "MeshCheckCode",
    "MeshConvCheck",
    "MeshConvErr",
    "MeshConvention",
    "MeshGeometry",
    "MeshVerifyErr",
    "MeshVerifyIssue",
    "SimData",
    "UserTopology",
    "check_mesh_convention",
    "convert_mesh",
    "enforce_mesh_convention",
    "extract_surf_between",
    "extract_surf_mesh",
    "extract_surface",
    "infer_mesh_convention",
    "verify_mesh",
]
