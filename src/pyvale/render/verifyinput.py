# ============================================================================
# pyvale: the python validation engine
# License: MIT
# Copyright (C) 2026 Sceptical Rabbit (Lloyd Fletcher)
# ============================================================================
"""Cheap, non-rendering validation helpers."""

from collections.abc import Sequence

import numpy as np

from pyvale.dataio import SimData, check_mesh_convention

from .camera import Camera
from .errors import RenderInputError, ValidationIssue
from .light import Light
from .mesh import Mesh3D


def mesh_convention_issues(
    coords: np.ndarray,
    connectivity: np.ndarray,
    path: str,
) -> tuple[ValidationIssue, ...]:
    """Return shared-convention issues for a render surface mesh.

    Two-dimensional coordinates are padded onto the XY plane before invoking
    the common DataIO checker. This keeps planar renderers aligned with the
    same counter-clockwise, right-handed convention as 3D renderers.
    """
    coords_array = np.asarray(coords)
    connectivity_array = np.asarray(connectivity)
    if coords_array.ndim != 2 or coords_array.shape[1] not in (2, 3):
        return tuple()
    if connectivity_array.ndim != 2 or connectivity_array.size == 0:
        return tuple()

    if coords_array.shape[1] == 2:
        coords_array = np.pad(coords_array, ((0, 0), (0, 1)))

    try:
        report = check_mesh_convention(SimData(
            coords=coords_array,
            connect={"connect1": connectivity_array},
        ))
    except (IndexError, NotImplementedError, TypeError, ValueError) as error:
        return (
            ValidationIssue(
                path + ".connectivity",
                "CONVENTION",
                f"Could not check the shared mesh convention: {error}",
            ),
        )

    if not report:
        return tuple()

    failed = "; ".join(
        f"{table}: {', '.join(code.value for code in codes)}"
        for table, codes in report.items()
    )
    return (
        ValidationIssue(
            path + ".connectivity",
            "CONVENTION",
            "Mesh must follow the shared Riley/VTK convention: " + failed,
        ),
    )


def verify_scene_3d(
    meshes: Sequence[Mesh3D],
    cameras: Sequence[Camera],
    lights: Sequence[Light] | None,
) -> tuple[ValidationIssue, ...]:
    """Check common three-dimensional scene constraints cheaply.

    Parameters
    ----------
    meshes : Sequence[Mesh3D]
        Surface meshes in the requested scene.
    cameras : Sequence[Camera]
        Cameras in the requested scene.
    lights : Sequence[Light] or None
        Explicit scene lights, if any.

    Returns
    -------
    tuple[ValidationIssue, ...]
        Every detected issue. An empty tuple denotes a valid common scene.

    Notes
    -----
    This routine performs no scene construction, image allocation, or backend
    calculation. Backends must add their own capability checks.
    """
    issues: list[ValidationIssue] = []

    if not meshes:
        issues.append(ValidationIssue(
            "meshes",
            "EMPTY",
            "At least one mesh is required.",
        ))

    if not cameras:
        issues.append(ValidationIssue(
            "cameras",
            "EMPTY",
            "At least one camera is required.",
        ))

    for mesh_index, mesh in enumerate(meshes):
        path = f"meshes[{mesh_index}]"

        if mesh.coords.ndim != 2 or mesh.coords.shape[1] != 3:
            issues.append(ValidationIssue(
                path + ".coords", "SHAPE", "Expected shape (nodes, 3).",
            ))
        elif not np.isfinite(mesh.coords).all():
            issues.append(ValidationIssue(
                path + ".coords",
                "FINITE",
                "Values must be finite.",
            ))

        if mesh.connectivity.ndim != 2 or mesh.connectivity.shape[0] == 0:
            issues.append(ValidationIssue(
                path + ".connectivity",
                "SHAPE",
                "Expected a non-empty rank-2 array.",
            ))
        elif (
            mesh.coords.ndim == 2
            and np.any(mesh.connectivity >= len(mesh.coords))
        ):
            issues.append(ValidationIssue(
                path + ".connectivity",
                "INDEX",
                "Indices exceed node count.",
            ))
        else:
            issues.extend(mesh_convention_issues(
                mesh.coords,
                mesh.connectivity,
                path,
            ))

        if mesh.displacements is not None:
            expected = (mesh.displacements.shape[0], mesh.coords.shape[0], 3)

            if mesh.displacements.shape != expected:
                issues.append(ValidationIssue(
                    path + ".displacements",
                    "SHAPE",
                    "Expected shape (frames, nodes, 3).",
                ))
            elif not np.isfinite(mesh.displacements).all():
                issues.append(ValidationIssue(
                    path + ".displacements",
                    "FINITE",
                    "Values must be finite.",
                ))

    for camera_index, camera in enumerate(cameras):
        path = f"cameras[{camera_index}]"

        if camera.pixels_num.shape != (2,) or np.any(camera.pixels_num <= 0):
            issues.append(ValidationIssue(
                path + ".pixels_num",
                "VALUE",
                "Expected two positive counts.",
            ))

        if (
            camera.pixels_size.shape != (2,)
            or np.any(camera.pixels_size <= 0.0)
        ):
            issues.append(ValidationIssue(
                path + ".pixels_size",
                "VALUE",
                "Expected two positive sizes.",
            ))

        if camera.focal_length <= 0.0 or camera.sub_sample <= 0:
            issues.append(ValidationIssue(
                path,
                "VALUE",
                "Focal length and sub-sampling must be positive.",
            ))

        vectors = (
            ("pos_world", camera.pos_world),
            ("roi_cent_world", camera.roi_cent_world),
        )
        for value_name, value in vectors:
            if value.shape != (3,) or not np.isfinite(value).all():
                issues.append(ValidationIssue(
                    path + "." + value_name,
                    "VALUE",
                    "Expected three finite values.",
                ))

    if lights is not None:
        for light_index, light in enumerate(lights):
            if not np.isfinite(light.intensity) or light.intensity < 0.0:
                issues.append(ValidationIssue(
                    f"lights[{light_index}].intensity", "VALUE",
                    "Expected a non-negative finite value.",
                ))
            if (not np.isfinite(light.shadow_soft_size)
                    or light.shadow_soft_size < 0.0):
                issues.append(ValidationIssue(
                    f"lights[{light_index}].shadow_soft_size", "VALUE",
                    "Expected a non-negative finite value.",
                ))

    return tuple(issues)


def raise_if_issues(issues: tuple[ValidationIssue, ...]) -> None:
    """Raise an aggregated error when validation detected issues.

    Parameters
    ----------
    issues : tuple[ValidationIssue, ...]
        Issues returned by a verification routine.

    Raises
    ------
    RenderInputError
        If ``issues`` is not empty.
    """
    if issues:
        raise RenderInputError(issues)


__all__ = ["mesh_convention_issues", "raise_if_issues", "verify_scene_3d"]
