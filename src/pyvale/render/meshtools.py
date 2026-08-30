# ==============================================================================
# pyvale: the python validation engine
# License: MIT
# Copyright (C) 2026 Sceptical Rabbit (Lloyd Fletcher)
# ==============================================================================
"""Conversion and geometric transformations for render surface meshes."""

from collections.abc import Sequence
from dataclasses import replace

import numpy as np
from scipy.spatial.transform import Rotation

from pyvale.dataio.meshconv import (
    enforce_mesh_convention,
    extract_surf_mesh,
    is_mesh_2d,
)
from pyvale.dataio.simdata import SimData

from .mesh import EElementType, Mesh3D


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
        Backend owned material or shader definition for the mesh.
    displacement_keys : Sequence[str] or None, optional
        Names of the three displacement components. ``None`` omits motion.

    Returns
    -------
    Mesh3D
        Renderer independent surface mesh data.
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


def mesh_bounds(mesh: Mesh3D) -> tuple[np.ndarray, np.ndarray]:
    """Calculate the axis aligned bounding box of a mesh.

    Parameters
    ----------
    mesh : Mesh3D
        Mesh to query.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        Tuple of ``(coord_min, coord_max)`` arrays, each with shape ``(3,)``
        and dtype ``float64`` representing (X, Y, Z) minimum and maximum
        extents.
    """
    coords = np.asarray(mesh.coords, dtype=np.float64)

    if coords.size == 0:
        return np.zeros(3), np.zeros(3)

    return np.min(coords, axis=0), np.max(coords, axis=0)


def mesh_center(mesh: Mesh3D) -> np.ndarray:
    """Calculate the center of the axis aligned bounding box of a mesh.

    Parameters
    ----------
    mesh : Mesh3D
        Mesh to query.

    Returns
    -------
    np.ndarray
        Midpoint of the bounding box with shape ``(3,)`` and dtype
        ``float64`` representing (X, Y, Z) center coordinates.
    """
    lower, upper = mesh_bounds(mesh)
    return 0.5 * (lower + upper)


def mesh_translate(mesh: Mesh3D, translation: np.ndarray) -> Mesh3D:
    """Translate a mesh by an offset vector without mutating the original.

    Note that translation shifts nodal coordinates but preserves nodal
    displacement vectors unchanged.

    Parameters
    ----------
    mesh : Mesh3D
        Source surface mesh to translate.
    translation : np.ndarray
        Translation vector array with shape ``(3,)`` or ``(2,)`` and dtype
        ``float64`` representing (dX, dY, dZ) offsets.

    Returns
    -------
    Mesh3D
        New translated mesh instance.
    """
    trans = np.asarray(translation, dtype=np.float64)
    new_coords = np.asarray(mesh.coords, dtype=np.float64) + trans
    return replace(mesh, coords=new_coords)


def mesh_rotate(
    mesh: Mesh3D,
    rotation: Rotation,
    pivot: np.ndarray | None = None,
) -> Mesh3D:
    """Rotate a mesh around a pivot point.

    Displacement vectors are rotated as vector fields to match coordinate
    reorientation.

    Parameters
    ----------
    mesh : Mesh3D
        Source surface mesh to rotate.
    rotation : scipy.spatial.transform.Rotation
        Spatial rotation to apply.
    pivot : np.ndarray or None, optional
        Pivot center point array with shape ``(3,)`` and dtype ``float64``.
        If ``None``, defaults to the origin ``(0, 0, 0)``.

    Returns
    -------
    Mesh3D
        New rotated mesh instance.
    """
    coords = np.asarray(mesh.coords, dtype=np.float64)
    p_vec = (
        np.zeros(3, dtype=np.float64)
        if pivot is None
        else np.asarray(pivot, dtype=np.float64)
    )

    rel_coords = coords - p_vec
    new_coords = rotation.apply(rel_coords) + p_vec

    new_displacements = None
    if getattr(mesh, "displacements", None) is not None:
        disp = np.asarray(mesh.displacements, dtype=np.float64)
        shape = disp.shape
        flat_disp = disp.reshape(-1, 3)
        rot_disp = rotation.apply(flat_disp)
        new_displacements = rot_disp.reshape(shape)

    return replace(mesh, coords=new_coords, displacements=new_displacements)


def mesh_scale(
    mesh: Mesh3D,
    scale: float | np.ndarray,
    pivot: np.ndarray | None = None,
) -> Mesh3D:
    """Scale a mesh relative to a pivot point.

    Parameters
    ----------
    mesh : Mesh3D
        Source surface mesh to scale.
    scale : float or np.ndarray
        Uniform scale factor as a float or per axis scale array with shape
        ``(3,)`` and dtype ``float64`` for (sX, sY, sZ) scaling.
    pivot : np.ndarray or None, optional
        Pivot center point array with shape ``(3,)`` and dtype ``float64``.
        If ``None``, defaults to the origin ``(0, 0, 0)``.

    Returns
    -------
    Mesh3D
        New scaled mesh instance.
    """
    coords = np.asarray(mesh.coords, dtype=np.float64)
    p_vec = (
        np.zeros(3, dtype=np.float64)
        if pivot is None
        else np.asarray(pivot, dtype=np.float64)
    )
    s_vec = np.broadcast_to(np.asarray(scale, dtype=np.float64), (3,))

    new_coords = (coords - p_vec) * s_vec + p_vec

    new_displacements = None
    if mesh.displacements is not None:
        new_displacements = mesh.displacements * s_vec

    return replace(
        mesh,
        coords=new_coords,
        displacements=new_displacements,
    )


def mesh_transform(
    mesh: Mesh3D,
    translation: np.ndarray | None = None,
    rotation: Rotation | None = None,
    scale: float | np.ndarray | None = None,
    pivot: np.ndarray | None = None,
) -> Mesh3D:
    """Apply affine scaling, rotation, and translation in canonical order.

    Order: 1. Scale about pivot, 2. Rotate about pivot, 3. Translate.

    Parameters
    ----------
    mesh : Mesh3D
        Source surface mesh to transform.
    translation : np.ndarray or None, optional
        Translation vector with shape ``(3,)`` and dtype ``float64``.
    rotation : scipy.spatial.transform.Rotation or None, optional
        Rotation to apply.
    scale : float or np.ndarray or None, optional
        Scaling factor or per axis scale array with shape ``(3,)``.
    pivot : np.ndarray or None, optional
        Pivot point array with shape ``(3,)`` and dtype ``float64``.

    Returns
    -------
    Mesh3D
        New transformed mesh instance.
    """
    transformed = mesh

    if scale is not None:
        transformed = mesh_scale(transformed, scale, pivot=pivot)

    if rotation is not None:
        transformed = mesh_rotate(transformed, rotation, pivot=pivot)

    if translation is not None:
        transformed = mesh_translate(transformed, translation)

    return transformed


def mesh_center_at(
    mesh: Mesh3D,
    target: np.ndarray = np.array((0.0, 0.0, 0.0)),
) -> Mesh3D:
    """Translate a mesh so its bounding box center lies at ``target``.

    Parameters
    ----------
    mesh : Mesh3D
        Source surface mesh to re centre.
    target : np.ndarray, optional
        Target center point array with shape ``(3,)`` and dtype ``float64``
        representing (X, Y, Z) coordinates. Defaults to ``(0, 0, 0)``.

    Returns
    -------
    Mesh3D
        New translated mesh instance.
    """
    current_center = mesh_center(mesh)
    target_vec = np.asarray(target, dtype=np.float64)[:3]
    delta = target_vec - current_center

    return mesh_translate(mesh, delta)


def evenly_spaced_frame_indices(
    total_frames: int,
    num_samples: int,
) -> np.ndarray:
    """Return sample indices evenly distributed from 0 to total_frames - 1.

    Parameters
    ----------
    total_frames : int
        Total number of frames available.
    num_samples : int
        Desired number of evenly distributed frame samples.

    Returns
    -------
    np.ndarray
        Array of integer frame indices with shape ``(N,)`` and dtype ``intp``,
        where ``N <= num_samples``.
    """
    if total_frames <= 0 or num_samples <= 0:
        return np.empty(0, dtype=np.intp)

    selected_num = min(total_frames, num_samples)
    if selected_num == 1:
        return np.array([0], dtype=np.intp)

    return np.array(
        [
            frame * (total_frames - 1) // (selected_num - 1)
            for frame in range(selected_num)
        ],
        dtype=np.intp,
    )


def first_last_frame_indices(total_frames: int) -> np.ndarray:
    """Return indices for the first and last frame.

    Parameters
    ----------
    total_frames : int
        Total number of available frames.

    Returns
    -------
    np.ndarray
        Array of frame indices with shape ``(2,)`` (or ``(1,)``/``(0,)``) and
        dtype ``intp``.
    """
    if total_frames <= 0:
        return np.empty(0, dtype=np.intp)

    if total_frames == 1:
        return np.array([0], dtype=np.intp)

    return np.array([0, total_frames - 1], dtype=np.intp)


def select_frames(
    frames: np.ndarray,
    indices: Sequence[int] | np.ndarray,
) -> np.ndarray:
    """Index a leading frame dimension with integer indices.

    Parameters
    ----------
    frames : np.ndarray
        Array whose leading dimension is frames, e.g. shape
        ``(num_frames, ...)``.
    indices : Sequence[int] or np.ndarray
        Indices to slice along the leading dimension.

    Returns
    -------
    np.ndarray
        Sliced array containing only the selected frames.
    """
    idx = np.asarray(indices, dtype=np.intp)

    return frames[idx]


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


def _element_type_from_nodes(nodes_per_element: int) -> EElementType:
    """Map a connectivity width to the corresponding render topology."""
    mapping = {
        3: EElementType.TRI3,
        6: EElementType.TRI6,
        4: EElementType.QUAD4,
        8: EElementType.QUAD8,
        9: EElementType.QUAD9,
    }

    if nodes_per_element not in mapping:
        raise ValueError(
            f"Unsupported surface connectivity with {nodes_per_element} "
            "nodes per element.",
        )

    return mapping[nodes_per_element]


def _displacements_from_simdata(
    sim_data: SimData,
    displacement_keys: Sequence[str] | None,
) -> np.ndarray | None:
    """Extract nodal displacement fields into renderer array order."""
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
            f"Missing displacement field {error.args[0]!r}."
        ) from error

    if any(field.ndim != 2 for field in fields):
        raise ValueError(
            "Displacement fields must have shape (nodes, frames)."
        )

    displacements = np.stack(fields, axis=2).transpose(1, 0, 2)

    if displacements.shape[2] == 2:
        displacements = np.pad(displacements, ((0, 0), (0, 0), (0, 1)))

    return np.ascontiguousarray(displacements)


__all__ = [
    "evenly_spaced_frame_indices",
    "first_last_frame_indices",
    "mesh3d_from_simdata",
    "mesh_bounds",
    "mesh_center",
    "mesh_center_at",
    "mesh_rotate",
    "mesh_scale",
    "mesh_transform",
    "mesh_translate",
    "select_frames",
]
