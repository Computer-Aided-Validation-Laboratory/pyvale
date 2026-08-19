# ============================================================================
# pyvale: the python validation engine
# License: MIT
# Copyright (C) 2026 Sceptical Rabbit (Lloyd Fletcher)
# ============================================================================
"""Riley adapter and Riley-owned shader specifications."""

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .capabilities import RenderCapabilities
from .camera import Camera
from .errors import RenderInputError, ValidationIssue
from .light import Light
from .mesh import EElementType, Mesh
from .renderer3d import IRenderer3D
from .result import RenderResult
from .verifyinput import raise_if_issues, verify_scene_3d

try:
    import riley as _riley
except ImportError:  # pragma: no cover - exercised by installed-package checks
    _riley = None


@dataclass(frozen=True, slots=True)
class TextureShader:
    """Riley texture shader with node UV coordinates."""

    uvs: np.ndarray
    texture: np.ndarray


@dataclass(frozen=True, slots=True)
class NodalFieldShader:
    """Riley scalar nodal field shader with ``(frames, nodes, fields)`` data."""

    values: np.ndarray


@dataclass(frozen=True, slots=True)
class FunctionShader:
    """Riley built-in analytic shader configuration."""

    builtin: int
    coord_mode: int
    params: object | None = None
    uvs: np.ndarray | None = None


@dataclass(frozen=True, slots=True)
class _RileyPlan:
    meshes: tuple[Mesh, ...]
    cameras: tuple[Camera, ...]


class Riley(IRenderer3D):
    """Render a validated pyvale scene through an actual Riley configuration."""

    capabilities = RenderCapabilities(
        element_types=frozenset(EElementType),
        supports_lights=False,
        supports_camera_distortion=True,
        supports_psf=True,
    )

    def __init__(
        self,
        riley_config: object,
        output_dir: Path | None = None,
    ) -> None:
        """Create an adapter around a genuine ``riley.RasterConfig`` object."""
        self.riley_config = riley_config
        self.output_dir = output_dir

    def verify_input(
        self,
        meshes: Sequence[Mesh],
        cameras: Sequence[Camera],
        lights: Sequence[Light] | None = None,
    ) -> _RileyPlan:
        """Verify all common and Riley-specific inputs before conversion."""
        issues = list(verify_scene_3d(meshes, cameras, lights))
        if _riley is None:
            issues.append(ValidationIssue("riley", "UNAVAILABLE", "riley-raster is not installed."))
        elif not isinstance(self.riley_config, _riley.RasterConfig):
            issues.append(ValidationIssue("riley_config", "TYPE", "Expected riley.RasterConfig."))
        if lights:
            issues.append(ValidationIssue("lights", "UNSUPPORTED", "Riley does not support lights yet."))
        frame_count: int | None = None
        for mesh_index, mesh in enumerate(meshes):
            if mesh.element_type not in self.capabilities.element_types:
                issues.append(ValidationIssue(f"meshes[{mesh_index}].element_type", "UNSUPPORTED", "Unsupported Riley element type."))
            if mesh.displacements is not None:
                if frame_count is None:
                    frame_count = mesh.displacements.shape[0]
                elif frame_count != mesh.displacements.shape[0]:
                    issues.append(ValidationIssue(f"meshes[{mesh_index}].displacements", "FRAME_COUNT", "All deforming meshes require the same frame count."))
            shader = mesh.shader
            if not isinstance(shader, (TextureShader, NodalFieldShader, FunctionShader)):
                issues.append(ValidationIssue(f"meshes[{mesh_index}].shader", "OWNERSHIP", "Expected a render.riley shader."))
                continue
            if isinstance(shader, TextureShader):
                if shader.uvs.shape != (mesh.coords.shape[0], 2):
                    issues.append(ValidationIssue(f"meshes[{mesh_index}].shader.uvs", "SHAPE", "Expected shape (nodes, 2)."))
                if shader.texture.ndim not in (2, 3):
                    issues.append(ValidationIssue(f"meshes[{mesh_index}].shader.texture", "SHAPE", "Expected a grey or RGB image."))
            if isinstance(shader, NodalFieldShader):
                if shader.values.ndim != 3 or shader.values.shape[1] != mesh.coords.shape[0]:
                    issues.append(ValidationIssue(f"meshes[{mesh_index}].shader.values", "SHAPE", "Expected shape (frames, nodes, fields)."))
            if isinstance(shader, FunctionShader) and shader.uvs is not None:
                if shader.uvs.shape != (mesh.coords.shape[0], 2):
                    issues.append(ValidationIssue(f"meshes[{mesh_index}].shader.uvs", "SHAPE", "Expected shape (nodes, 2)."))
        raise_if_issues(tuple(issues))
        return _RileyPlan(tuple(meshes), tuple(cameras))

    def _render(self, render_plan: object) -> RenderResult:
        """Convert a verified plan and invoke Riley exactly once."""
        if not isinstance(render_plan, _RileyPlan):
            raise TypeError("Riley received an invalid render plan.")
        assert _riley is not None
        riley_meshes = [_mesh_to_riley(mesh) for mesh in render_plan.meshes]
        riley_cameras = [_camera_to_riley(camera) for camera in render_plan.cameras]
        output_dir = None if self.output_dir is None else str(self.output_dir)
        if self.output_dir is not None:
            self.output_dir.mkdir(parents=True, exist_ok=True)
        images = _riley.raster(
            riley_meshes, riley_cameras, self.riley_config, out_dir=output_dir,
        )
        if images is not None:
            images = np.ascontiguousarray(images.transpose(0, 1, 3, 4, 2))
        return RenderResult(images=images)


def _mesh_to_riley(mesh: Mesh) -> object:
    assert _riley is not None
    mesh_types = {
        EElementType.TRI3: _riley.MeshType.tri3,
        EElementType.TRI6: _riley.MeshType.tri6,
        EElementType.QUAD4: _riley.MeshType.quad4ibi,
        EElementType.QUAD8: _riley.MeshType.quad8,
        EElementType.QUAD9: _riley.MeshType.quad9,
    }
    common = {
        "mesh_type": mesh_types[mesh.element_type],
        "coords": mesh.coords,
        "connect": mesh.connectivity,
        "disp": mesh.displacements,
    }
    shader = mesh.shader
    if isinstance(shader, TextureShader):
        return _riley.Mesh(
            shader_type=_riley.ShaderType.tex,
            uvs=np.ascontiguousarray(shader.uvs, dtype=np.float64),
            texture=np.ascontiguousarray(shader.texture),
            **common,
        )
    if isinstance(shader, NodalFieldShader):
        return _riley.Mesh(
            shader_type=_riley.ShaderType.nodal,
            nodal_field=np.ascontiguousarray(shader.values, dtype=np.float64),
            **common,
        )
    assert isinstance(shader, FunctionShader)
    kwargs: dict[str, object] = {
        "shader_type": _riley.ShaderType.func,
        "func_shader_builtin": shader.builtin,
        "func_shader_coord_mode": shader.coord_mode,
        **common,
    }
    if shader.params is not None:
        kwargs["func_shader_params"] = shader.params
    if shader.uvs is not None:
        kwargs["uvs"] = np.ascontiguousarray(shader.uvs, dtype=np.float64)
    return _riley.Mesh(**kwargs)


def _camera_to_riley(camera: Camera) -> object:
    assert _riley is not None
    return _riley.Camera(
        pixels_num=tuple(int(value) for value in camera.pixels_num),
        pixels_size=tuple(float(value) for value in camera.pixels_size),
        pos_world=tuple(float(value) for value in camera.pos_world),
        rot_world=tuple(float(value) for value in camera.rot_world.as_euler("xyz")),
        roi_cent_world=tuple(float(value) for value in camera.roi_cent_world),
        focal_length=camera.focal_length,
        sub_sample=camera.sub_sample,
        distortion_model=camera.distortion_model,
        distortion_k1=camera.distortion_k1,
        distortion_k2=camera.distortion_k2,
        distortion_k3=camera.distortion_k3,
        distortion_k4=camera.distortion_k4,
        distortion_k5=camera.distortion_k5,
        distortion_k6=camera.distortion_k6,
        distortion_p1=camera.distortion_p1,
        distortion_p2=camera.distortion_p2,
        psf_type=camera.psf_type,
        psf_sigma_x=camera.psf_sigma_x,
        psf_sigma_y=camera.psf_sigma_y,
        psf_theta=camera.psf_theta,
        psf_support_rad=camera.psf_support_rad,
    )


__all__ = ["FunctionShader", "NodalFieldShader", "Riley", "TextureShader"]
