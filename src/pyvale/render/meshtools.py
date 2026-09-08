# ==============================================================================
# pyvale: the python validation engine
# License: MIT
# Copyright (C) 2026 Sceptical Rabbit (Lloyd Fletcher)
# ==============================================================================
"""Conversion and geometric transformations for render surface meshes."""

from collections.abc import Mapping, Sequence
from dataclasses import replace

import numpy as np
import riley
from scipy.spatial.transform import Rotation

from pyvale.dataio.simdata import SimData

from .mesh import Mesh3D


def meshes3d_from_simdata(
    sim_data: SimData,
    conventions: Mapping[str, riley.ConnectConvention],
    *,
    mesh_types: Mapping[str, riley.MeshType] | None = None,
    shaders: Mapping[str, object | None] | None = None,
    displacement_keys: Sequence[str] | None = None,
) -> dict[str, Mesh3D]:
    """Build one prepared surface mesh for each simulation mesh block.

    Parameters
    ----------
    sim_data : pyvale.dataio.SimData
        Simulation data containing coordinates, connectivity, and optionally
        nodal displacement fields.
    conventions : mapping of str to riley.ConnectConvention
        Explicit source connectivity convention for every block.
    mesh_types : mapping of str to riley.MeshType or None, optional
        Target Riley topology/implementation for each block. Natural surface
        types are used when omitted.
    shaders : mapping of str to object or None, optional
        Backend-owned shader for each block.
    displacement_keys : Sequence[str] or None, optional
        Names of the three displacement components. ``None`` omits motion.

    Returns
    -------
    dict of str to Mesh3D
        Prepared meshes keyed by input connectivity block.
    """
    if sim_data.coords is None or sim_data.connect is None:
        raise ValueError("SimData must provide coordinates and connectivity.")
    block_keys = set(sim_data.connect)
    if set(conventions) != block_keys:
        raise ValueError("Convention keys must match connectivity block keys.")
    if mesh_types is not None and set(mesh_types) != block_keys:
        raise ValueError("Mesh type keys must match connectivity block keys.")
    if shaders is not None and set(shaders) != block_keys:
        raise ValueError("Shader keys must match connectivity block keys.")

    coords = _coords3d(sim_data.coords)
    disp = _displacement_components_from_simdata(
        sim_data, displacement_keys, coords.shape[0]
    )
    meshes = {}
    for block_key, connectivity in sim_data.connect.items():
        convention = conventions[block_key]
        mesh_type = (
            _natural_mesh_type(convention.elem_type)
            if mesh_types is None
            else mesh_types[block_key]
        )
        conversion = riley.convert_mesh_for_render(
            convention, mesh_type, coords, connectivity
        )
        displacements = None
        if disp is not None:
            remapped = tuple(
                riley.remap_nodal_data(conversion, component)
                for component in disp
            )
            displacements = np.ascontiguousarray(
                np.stack(remapped, axis=2).transpose(1, 0, 2)
            )
        geometry = conversion.geometry
        meshes[block_key] = Mesh3D(
            element_type=geometry.elem_type,
            coords=geometry.coords,
            connectivity=geometry.connect,
            shader=None if shaders is None else shaders[block_key],
            displacements=displacements,
        )
    return meshes


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


def _coords3d(coords: np.ndarray) -> np.ndarray:
    """Return finite render coordinates padded to three dimensions."""
    coords_out = np.ascontiguousarray(coords, dtype=np.float64)
    if coords_out.ndim != 2 or coords_out.shape[1] not in (2, 3):
        raise ValueError("SimData coordinates must have two or three columns.")

    if coords_out.shape[1] == 2:
        coords_out = np.pad(coords_out, ((0, 0), (0, 1)))

    return coords_out


def _natural_mesh_type(elem_type: riley.EElemType) -> riley.MeshType:
    """Return the natural Riley renderer type for source geometry."""
    mapping = {
        riley.EElemType.TRI3: riley.MeshType.tri3,
        riley.EElemType.TRI6: riley.MeshType.tri6,
        riley.EElemType.TRI7: riley.MeshType.tri6,
        riley.EElemType.QUAD4: riley.MeshType.quad4newton,
        riley.EElemType.QUAD8: riley.MeshType.quad8,
        riley.EElemType.QUAD9: riley.MeshType.quad9,
        riley.EElemType.TET4: riley.MeshType.tri3,
        riley.EElemType.TET10: riley.MeshType.tri6,
        riley.EElemType.HEX8: riley.MeshType.quad4newton,
        riley.EElemType.HEX20: riley.MeshType.quad8,
        riley.EElemType.HEX27: riley.MeshType.quad9,
    }
    return mapping[elem_type]


def _displacement_components_from_simdata(
    sim_data: SimData,
    displacement_keys: Sequence[str] | None,
    nodes_num: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
    """Extract three source-indexed displacement components."""
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

    if any(field.shape[0] != nodes_num for field in fields):
        raise ValueError("Displacement node count must match coordinates.")
    if len(fields) == 2:
        fields.append(np.zeros_like(fields[0]))
    return tuple(np.ascontiguousarray(field) for field in fields)


__all__ = [
    "evenly_spaced_frame_indices",
    "first_last_frame_indices",
    "mesh_bounds",
    "mesh_center",
    "mesh_center_at",
    "mesh_rotate",
    "mesh_scale",
    "mesh_transform",
    "mesh_translate",
    "meshes3d_from_simdata",
    "select_frames",
]
