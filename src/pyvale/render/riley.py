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
from .mesh import EElementType, Mesh3D
from .renderer3d import IRenderer3D
from .result import RenderResult
from .rileyshader import (
    RileyFunctionShader,
    RileyNodalShader,
    RileyTextureShader,
)
from .scene import Scene3D
from .verifyinput import mesh_convention_issues, raise_if_issues


class Riley(IRenderer3D):
    """Render native Riley meshes through pyvale's common scene container.

    Riley owns its complete mesh and shader representation. Construct meshes
    with :class:`riley.Mesh`, including its texture, nodal-field, or analytic
    function shader settings, then place them in :class:`Scene3D`.

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

    def verify_input(self, scene: Scene3D) -> None:
        """Verify native Riley meshes and common cameras before rasterisation.

        Parameters
        ----------
        scene : Scene3D
            Scene containing native Riley meshes and no explicit lights.

        Raises
        ------
        RenderInputError
            If configuration, mesh, camera, or lighting input is unsupported.
        """
        issues: list[ValidationIssue] = []

        if not isinstance(scene, Scene3D):
            issues.append(
                ValidationIssue(
                    "scene",
                    "TYPE",
                    "Expected a Scene3D.",
                )
            )
            raise_if_issues(tuple(issues))
            return

        if not isinstance(self.config, riley.RasterConfig):
            issues.append(
                ValidationIssue(
                    "config",
                    "TYPE",
                    "Expected riley.RasterConfig.",
                )
            )

        if not scene.meshes:
            issues.append(
                ValidationIssue(
                    "scene.meshes",
                    "EMPTY",
                    "At least one mesh is required.",
                )
            )

        if not scene.cameras:
            issues.append(
                ValidationIssue(
                    "scene.cameras",
                    "EMPTY",
                    "At least one camera is required.",
                )
            )

        if scene.lights:
            issues.append(
                ValidationIssue(
                    "scene.lights",
                    "UNSUPPORTED",
                    "Riley does not support lights yet.",
                )
            )

        for mesh_index, mesh in enumerate(scene.meshes):
            if not isinstance(mesh, (Mesh3D, riley.Mesh)):
                issues.append(
                    ValidationIssue(
                        f"scene.meshes[{mesh_index}]",
                        "TYPE",
                        "Riley requires render.Mesh3D or riley.Mesh objects.",
                    )
                )
                continue

            if isinstance(mesh, Mesh3D) and not isinstance(
                mesh.shader,
                (RileyFunctionShader, RileyNodalShader, RileyTextureShader),
            ):
                issues.append(
                    ValidationIssue(
                        f"scene.meshes[{mesh_index}].shader",
                        "TYPE",
                        "Expected a Riley shader description.",
                    )
                )

            issues.extend(
                mesh_convention_issues(
                    mesh.coords,
                    mesh.connectivity
                    if isinstance(mesh, Mesh3D)
                    else mesh.connect,
                    f"scene.meshes[{mesh_index}]",
                )
            )

        issues.extend(_verify_cameras(scene.cameras))
        raise_if_issues(tuple(issues))

    def _render(self, scene: Scene3D) -> RenderResult:
        """Rasterise a previously validated scene exactly once.

        Parameters
        ----------
        scene : Scene3D
            Validated scene containing native Riley meshes.

        Returns
        -------
        RenderResult
            Riley images in ``(frame, camera, height, width, channel)`` order.
        """
        output_dir = None if self.output_dir is None else str(self.output_dir)

        if self.output_dir is not None:
            self.output_dir.mkdir(parents=True, exist_ok=True)

        meshes = [to_native_mesh(mesh) for mesh in scene.meshes]
        cameras = [
            to_native_camera(camera) for camera in scene.cameras
        ]
        images = riley.raster(
            meshes,
            cameras,
            self.config,
            out_dir=output_dir,
        )

        if images is not None:
            images = np.ascontiguousarray(images.transpose(1, 0, 3, 4, 2))

        return RenderResult(images=images)


def _verify_cameras(
    cameras: tuple[Camera | riley.Camera, ...],
) -> tuple[ValidationIssue, ...]:
    """Perform the common cheap camera checks without requiring common meshes."""

    issues: list[ValidationIssue] = []
    for index, camera in enumerate(cameras):
        path = f"scene.cameras[{index}]"
        if not isinstance(camera, (Camera, riley.Camera)):
            issues.append(
                ValidationIssue(
                    path,
                    "TYPE",
                    "Expected a render.Camera or riley.Camera.",
                )
            )
            continue

        if not isinstance(camera, Camera):
            continue

        if camera.pixels_num.shape != (2,) or np.any(camera.pixels_num <= 0):
            issues.append(
                ValidationIssue(
                    path + ".pixels_num",
                    "VALUE",
                    "Expected two positive counts.",
                )
            )

        if camera.pixels_size.shape != (2,) or np.any(
            camera.pixels_size <= 0.0
        ):
            issues.append(
                ValidationIssue(
                    path + ".pixels_size",
                    "VALUE",
                    "Expected two positive sizes.",
                )
            )

        if camera.focal_length <= 0.0 or camera.subsample <= 0:
            issues.append(
                ValidationIssue(
                    path,
                    "VALUE",
                    "Focal length and sub-sampling must be positive.",
                )
            )

    return tuple(issues)


def to_native_camera(camera: Camera | riley.Camera) -> riley.Camera:
    """Convert one common perspective camera to a Riley camera.

    Native :class:`riley.Camera` instances are returned unchanged so scenes
    may mix cameras built through the render API with cameras loaded
    directly from Riley.
    """
    if isinstance(camera, riley.Camera):
        return camera
    return riley.Camera(
        pixels_num=tuple(int(value) for value in camera.pixels_num),
        pixels_size=tuple(float(value) for value in camera.pixels_size),
        pos_world=tuple(float(value) for value in camera.pos_world),
        rot_world=tuple(
            float(value) for value in camera.rot_world.as_euler("xyz")
        ),
        roi_cent_world=tuple(float(value) for value in camera.roi_cent_world),
        focal_length=camera.focal_length,
        sub_sample=camera.subsample,
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


_RILEY_MESH_TYPES = {
    EElementType.TRI3: riley.MeshType.tri3,
    EElementType.TRI6: riley.MeshType.tri6,
    EElementType.QUAD4: riley.MeshType.quad4newton,
    EElementType.QUAD8: riley.MeshType.quad8,
    EElementType.QUAD9: riley.MeshType.quad9,
}


def to_native_mesh(mesh: Mesh3D | riley.Mesh) -> riley.Mesh:
    """Convert one common mesh to a native Riley mesh.

    Native :class:`riley.Mesh` instances are returned unchanged so scenes
    may mix meshes built through the render API with meshes built directly
    through Riley.
    """
    if isinstance(mesh, riley.Mesh):
        return mesh

    common = {
        "mesh_type": _RILEY_MESH_TYPES[mesh.element_type],
        "coords": mesh.coords,
        "connect": mesh.connectivity,
        "disp": mesh.displacements,
    }
    shader = mesh.shader

    if isinstance(shader, RileyTextureShader):
        return riley.Mesh(
            shader_type=riley.ShaderType.tex,
            uvs=shader.uvs,
            texture=shader.texture,
            sample=shader.sample,
            sample_mode=shader.sample_mode,
            bits=shader.bits,
            scaling_type=shader.scaling,
            **common,
        )

    if isinstance(shader, RileyNodalShader):
        return riley.Mesh(
            shader_type=riley.ShaderType.nodal,
            nodal_field=shader.field,
            bits=shader.bits,
            scaling_type=shader.scaling,
            scale_over=shader.scale_over,
            **common,
        )

    if isinstance(shader, RileyFunctionShader):
        return riley.Mesh(
            shader_type=riley.ShaderType.func,
            uvs=shader.uvs,
            func_shader_builtin=shader.builtin,
            func_shader_coord_mode=shader.coord_mode,
            func_shader_params=shader.parameters,
            bits=shader.bits,
            scaling_type=shader.scaling,
            **common,
        )

    raise TypeError(
        "Riley Mesh3D.shader must be RileyFunctionShader, "
        "RileyTextureShader, or RileyNodalShader."
    )


__all__ = ["Riley", "to_native_camera", "to_native_mesh"]
