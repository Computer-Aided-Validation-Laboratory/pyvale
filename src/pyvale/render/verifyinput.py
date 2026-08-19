# ============================================================================
# pyvale: the python validation engine
# License: MIT
# Copyright (C) 2026 Sceptical Rabbit (Lloyd Fletcher)
# ============================================================================
"""Cheap, non-rendering validation helpers."""

from collections.abc import Sequence

import numpy as np

from .camera import Camera
from .errors import RenderInputError, ValidationIssue
from .light import Light
from .mesh import Mesh


def verify_scene_3d(
    meshes: Sequence[Mesh],
    cameras: Sequence[Camera],
    lights: Sequence[Light] | None,
) -> tuple[ValidationIssue, ...]:
    """Return all common 3D input issues without allocating render buffers."""
    issues: list[ValidationIssue] = []
    if not meshes:
        issues.append(ValidationIssue("meshes", "EMPTY", "At least one mesh is required."))
    if not cameras:
        issues.append(ValidationIssue("cameras", "EMPTY", "At least one camera is required."))
    for mesh_index, mesh in enumerate(meshes):
        path = f"meshes[{mesh_index}]"
        if mesh.coords.ndim != 2 or mesh.coords.shape[1] != 3:
            issues.append(ValidationIssue(path + ".coords", "SHAPE", "Expected shape (nodes, 3)."))
        elif not np.isfinite(mesh.coords).all():
            issues.append(ValidationIssue(path + ".coords", "FINITE", "Values must be finite."))
        if mesh.connectivity.ndim != 2 or mesh.connectivity.shape[0] == 0:
            issues.append(ValidationIssue(path + ".connectivity", "SHAPE", "Expected a non-empty rank-2 array."))
        elif mesh.coords.ndim == 2 and np.any(mesh.connectivity >= len(mesh.coords)):
            issues.append(ValidationIssue(path + ".connectivity", "INDEX", "Indices exceed node count."))
        if mesh.displacements is not None:
            expected = (mesh.displacements.shape[0], mesh.coords.shape[0], 3)
            if mesh.displacements.shape != expected:
                issues.append(ValidationIssue(path + ".displacements", "SHAPE", "Expected shape (frames, nodes, 3)."))
            elif not np.isfinite(mesh.displacements).all():
                issues.append(ValidationIssue(path + ".displacements", "FINITE", "Values must be finite."))
    for camera_index, camera in enumerate(cameras):
        path = f"cameras[{camera_index}]"
        if camera.pixels_num.shape != (2,) or np.any(camera.pixels_num <= 0):
            issues.append(ValidationIssue(path + ".pixels_num", "VALUE", "Expected two positive counts."))
        if camera.pixels_size.shape != (2,) or np.any(camera.pixels_size <= 0.0):
            issues.append(ValidationIssue(path + ".pixels_size", "VALUE", "Expected two positive sizes."))
        if camera.focal_length <= 0.0 or camera.sub_sample <= 0:
            issues.append(ValidationIssue(path, "VALUE", "Focal length and sub-sampling must be positive."))
        for value_name, value in (("pos_world", camera.pos_world), ("roi_cent_world", camera.roi_cent_world)):
            if value.shape != (3,) or not np.isfinite(value).all():
                issues.append(ValidationIssue(path + "." + value_name, "VALUE", "Expected three finite values."))
    if lights is not None:
        for light_index, light in enumerate(lights):
            if not np.isfinite(light.intensity) or light.intensity < 0.0:
                issues.append(ValidationIssue(f"lights[{light_index}].intensity", "VALUE", "Expected a non-negative finite value."))
    return tuple(issues)


def raise_if_issues(issues: tuple[ValidationIssue, ...]) -> None:
    """Raise one aggregated error if validation found any issues."""
    if issues:
        raise RenderInputError(issues)


__all__ = ["raise_if_issues", "verify_scene_3d"]
