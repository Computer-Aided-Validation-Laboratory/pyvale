# ============================================================================
# pyvale: the python validation engine
# License: MIT
# Copyright (C) 2026 Sceptical Rabbit (Lloyd Fletcher)
# ============================================================================
"""Riley renderer adapter using Riley's native public mesh and shader API."""

from pathlib import Path

import numpy as np
import riley

from .camera import Camera
from .capabilities import RenderCapabilities
from .errors import ValidationIssue
from .light import Light
from .mesh import EElementType
from .renderer3d import IRenderer3D
from .result import RenderResult
from .scene import RenderScene
from .verifyinput import mesh_convention_issues, raise_if_issues


class Riley(IRenderer3D):
    """Render native Riley meshes through pyvale's common scene container.

    Riley owns its complete mesh and shader representation. Construct meshes
    with :class:`riley.Mesh`, including its texture, nodal-field, or analytic
    function shader settings, then place them in :class:`RenderScene`.

    Parameters
    ----------
    config : riley.RasterConfig
        Actual Riley raster configuration passed directly to :func:`riley.raster`.
    output_dir : pathlib.Path or None, optional
        Directory to which Riley writes requested output. ``None`` preserves
        Riley's default output behaviour.
    """

    capabilities = RenderCapabilities(
        element_types=frozenset(EElementType),
        supports_lights=False,
        supports_camera_distortion=True,
        supports_psf=True,
    )

    def __init__(
        self,
        config: riley.RasterConfig,
        output_dir: Path | None = None,
    ) -> None:
        """Create a renderer around a concrete Riley raster configuration."""
        self.config = config
        self.output_dir = output_dir

    def verify_input(self, scene: RenderScene) -> None:
        """Verify native Riley meshes and common cameras before rasterisation.

        Parameters
        ----------
        scene : RenderScene
            Scene containing native Riley meshes and no explicit lights.

        Raises
        ------
        RenderInputError
            If configuration, mesh, camera, or lighting input is unsupported.
        """
        issues: list[ValidationIssue] = []
        if not isinstance(scene, RenderScene):
            issues.append(ValidationIssue(
                "scene", "TYPE", "Expected a RenderScene.",
            ))
            raise_if_issues(tuple(issues))
            return
        if not isinstance(self.config, riley.RasterConfig):
            issues.append(ValidationIssue(
                "config", "TYPE", "Expected riley.RasterConfig.",
            ))
        if not scene.meshes:
            issues.append(ValidationIssue(
                "scene.meshes", "EMPTY", "At least one mesh is required.",
            ))
        if not scene.cameras:
            issues.append(ValidationIssue(
                "scene.cameras", "EMPTY", "At least one camera is required.",
            ))
        if scene.lights:
            issues.append(ValidationIssue(
                "scene.lights", "UNSUPPORTED", "Riley does not support lights yet.",
            ))
        for mesh_index, mesh in enumerate(scene.meshes):
            if not isinstance(mesh, riley.Mesh):
                issues.append(ValidationIssue(
                    f"scene.meshes[{mesh_index}]", "TYPE",
                    "Riley requires native riley.Mesh objects.",
                ))
                continue
            issues.extend(mesh_convention_issues(
                mesh.coords,
                mesh.connect,
                f"scene.meshes[{mesh_index}]",
            ))
        issues.extend(_verify_cameras(scene.cameras))
        raise_if_issues(tuple(issues))

    def _render(self, scene: RenderScene) -> RenderResult:
        """Rasterise a previously validated scene exactly once.

        Parameters
        ----------
        scene : RenderScene
            Validated scene containing native Riley meshes.

        Returns
        -------
        RenderResult
            Riley images in ``(frame, camera, height, width, channel)`` order.
        """
        output_dir = None if self.output_dir is None else str(self.output_dir)
        if self.output_dir is not None:
            self.output_dir.mkdir(parents=True, exist_ok=True)
        images = riley.raster(
            list(scene.meshes),
            [_camera_to_riley(camera) for camera in scene.cameras],
            self.config,
            out_dir=output_dir,
        )
        if images is not None:
            images = np.ascontiguousarray(images.transpose(0, 1, 3, 4, 2))
        return RenderResult(images=images)


def _verify_cameras(cameras: tuple[Camera, ...]) -> tuple[ValidationIssue, ...]:
    """Perform the common cheap camera checks without requiring common meshes."""

    issues: list[ValidationIssue] = []
    for index, camera in enumerate(cameras):

        path = f"scene.cameras[{index}]"
        if not isinstance(camera, Camera):
            issues.append(ValidationIssue(
                path, "TYPE", "Expected a render.Camera.",
            ))
            continue

        if camera.pixels_num.shape != (2,) or np.any(camera.pixels_num <= 0):
            issues.append(ValidationIssue(
                path + ".pixels_num", "VALUE", "Expected two positive counts.",
            ))

        if camera.pixels_size.shape != (2,) or np.any(camera.pixels_size <= 0.0):
            issues.append(ValidationIssue(
                path + ".pixels_size", "VALUE", "Expected two positive sizes.",
            ))

        if camera.focal_length <= 0.0 or camera.sub_sample <= 0:
            issues.append(ValidationIssue(
                path, "VALUE", "Focal length and sub-sampling must be positive.",
            ))

    return tuple(issues)


def _camera_to_riley(camera: Camera) -> riley.Camera:
    """Convert one common perspective camera to a Riley camera."""
    return riley.Camera(
        pixels_num=tuple(int(value) for value in camera.pixels_num),
        pixels_size=tuple(float(value) for value in camera.pixels_size),
        pos_world=tuple(float(value) for value in camera.pos_world),
        rot_world=tuple(float(value) for value in camera.rot_world.as_euler("xyz")),
        roi_cent_world=tuple(float(value) for value in camera.roi_cent_world),
        focal_length=camera.focal_length,
        sub_sample=camera.sub_sample,
        distortion_model=int(camera.distortion_model),
        distortion_k1=camera.distortion_k1,
        distortion_k2=camera.distortion_k2,
        distortion_k3=camera.distortion_k3,
        distortion_k4=camera.distortion_k4,
        distortion_k5=camera.distortion_k5,
        distortion_k6=camera.distortion_k6,
        distortion_p1=camera.distortion_p1,
        distortion_p2=camera.distortion_p2,
        psf_type=int(camera.psf_type),
        psf_sigma_x=camera.psf_sigma_x,
        psf_sigma_y=camera.psf_sigma_y,
        psf_theta=camera.psf_theta,
        psf_support_rad=camera.psf_support_rad,
    )


__all__ = ["Riley"]
