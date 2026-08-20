"""Unified Blender renderer adapter."""

import importlib
from pathlib import Path
import sys

import numpy as np
from scipy.spatial.transform import Rotation

from pyvale.sensorsim.simtools import centre_mesh_nodes

from ..camera import Camera
from ..errors import ValidationIssue
from ..light import ELightType, Light
from ..mesh import Mesh
from ..renderer3d import IRenderer3D
from ..result import RenderResult
from ..scene import RenderScene
from ..verifyinput import raise_if_issues, verify_scene_3d
from .config import BlenderConfig, EBlenderEngine
from .shader import BlenderImageShader, BlenderTextureShader


class Blender(IRenderer3D):
    """Render common meshes, cameras, lights, and deformation frames in Blender."""

    def __init__(self, config: BlenderConfig) -> None:
        """Store configuration used by subsequent requests."""
        self.config = config

    def verify_input(
        self,
        scene: RenderScene,
    ) -> None:
        """Validate a complete Blender request before scene construction."""
        if not isinstance(scene, RenderScene):
            raise TypeError("Blender requires a RenderScene.")
        meshes = tuple(mesh for mesh in scene.meshes if isinstance(mesh, Mesh))
        issues = list(verify_scene_3d(meshes, scene.cameras, scene.lights))
        if not isinstance(self.config, BlenderConfig):
            issues.append(ValidationIssue("config", "TYPE", "Expected BlenderConfig."))
        elif (not isinstance(self.config.samples, int)
              or not isinstance(self.config.max_bounces, int)
              or not isinstance(self.config.threads, int)
              or self.config.samples <= 0 or self.config.max_bounces <= 0
              or self.config.threads <= 0):
            issues.append(ValidationIssue(
                "config", "VALUE",
                "Samples, bounces, and threads must be positive integers.",
            ))
        elif not isinstance(self.config.engine, EBlenderEngine):
            issues.append(ValidationIssue(
                "config.engine", "VALUE", "Unsupported Blender engine.",
            ))
        elif (not isinstance(self.config.render_deformed, bool)
              or not isinstance(self.config.save_images, bool)
              or not isinstance(self.config.save_scene, bool)):
            issues.append(ValidationIssue(
                "config", "TYPE", "Blender output controls must be booleans.",
            ))
        if len(scene.cameras) > 2:
            issues.append(ValidationIssue(
                "cameras", "COUNT",
                "Blender currently supports at most two cameras.",
            ))
        if sum(
            isinstance(mesh, Mesh) and mesh.displacements is not None
            for mesh in scene.meshes
        ) > 1:
            issues.append(ValidationIssue(
                "meshes", "DEFORMATION",
                "Only one deformable mesh is currently supported.",
            ))
        if scene.lights is not None and any(
            light.light_type not in ELightType for light in scene.lights
        ):
            issues.append(ValidationIssue(
                "lights", "TYPE", "Unsupported Blender light.",
            ))
        reason = _blender_unavailable_reason()
        if reason is not None:
            issues.append(ValidationIssue("blender", "UNAVAILABLE", reason))
        for mesh_index, mesh in enumerate(scene.meshes):
            if not isinstance(mesh, Mesh):
                issues.append(ValidationIssue(
                    f"scene.meshes[{mesh_index}]", "TYPE",
                    "Blender requires common render.Mesh objects.",
                ))
        raise_if_issues(tuple(issues))

    def _render(self, render_scene: RenderScene) -> RenderResult:
        """Construct and render one validated Blender scene."""
        blender_module = importlib.import_module("pyvale.blender")
        scene = blender_module.Scene()
        parts = [scene.add_part(mesh, 3) for mesh in render_scene.meshes]
        for mesh, part in zip(render_scene.meshes, parts):
            if isinstance(mesh.shader, BlenderTextureShader):
                scene.add_speckle(
                    part,
                    mesh.shader.image_path,
                    blender_module.MaterialData(
                        roughness=mesh.shader.material.roughness,
                        metallic=mesh.shader.material.metallic,
                        interpolant=mesh.shader.material.interpolant,
                    ),
                    mesh.shader.millimetres_per_pixel,
                )
            elif isinstance(mesh.shader, BlenderImageShader):
                material = blender_module.MaterialData(
                    roughness=mesh.shader.material.roughness,
                    metallic=mesh.shader.material.metallic,
                    interpolant=mesh.shader.material.interpolant,
                )
                blender_module.Tools.clear_material_nodes(part)
                blender_module.Tools.add_image_texture(
                    material, image_array=mesh.shader.image,
                )
                blender_module.Tools.uv_unwrap_part(
                    part, mesh.shader.millimetres_per_pixel,
                )
        for camera in render_scene.cameras:
            scene.add_camera(camera)
        for light in render_scene.lights or ():
            scene.add_light(_legacy_light(blender_module, light))
        self.config.output_dir.mkdir(parents=True, exist_ok=True)
        camera_data = (render_scene.cameras[0]
                       if len(render_scene.cameras) == 1
                       else render_scene.cameras)
        render_data = blender_module.RenderData(
            cam_data=camera_data,
            base_dir=self.config.output_dir,
            samples=self.config.samples,
            max_bounces=self.config.max_bounces,
            threads=self.config.threads,
            engine=blender_module.RenderEngine(self.config.engine.value),
        )
        deformable = next((mesh for mesh in render_scene.meshes
                           if mesh.displacements is not None), None)
        if not self.config.render_deformed:
            deformable = None
        if deformable is None:
            image = scene.render_single_image(
                render_data, stage_image=not self.config.save_images,
            )
            if self.config.save_scene:
                blender_module.Tools.save_blender_file(self.config.output_dir,
                                                       over_write=True)
            if image is None:
                return RenderResult(None, _image_paths(self.config.output_dir))
            return RenderResult(_normalise_images(np.asarray(image)))
        part = parts[render_scene.meshes.index(deformable)]
        deformation_mesh = Mesh(
            deformable.element_type,
            centre_mesh_nodes(deformable.coords.copy(), 3),
            deformable.connectivity,
            deformable.shader,
            deformable.displacements,
        )
        image = scene.render_deformed_images(
            deformation_mesh, 3, render_data,
            part, stage_image=not self.config.save_images,
        )
        if self.config.save_scene:
            blender_module.Tools.save_blender_file(self.config.output_dir,
                                                   over_write=True)
        if image is None:
            return RenderResult(None, _image_paths(self.config.output_dir))
        return RenderResult(_normalise_deformed_images(
            np.asarray(image), len(render_scene.cameras),
        ))


def _legacy_light(blender_module: object, light: Light) -> object:
    """Convert common light data to the legacy Blender scene light container."""
    light_type = blender_module.LightType(light.light_type.value.upper())
    direction = np.asarray(light.direction_world, dtype=np.float64)
    rotation = Rotation.identity()
    if np.linalg.norm(direction) > 0.0:
        rotation = Rotation.align_vectors(
            np.asarray(((0.0, 0.0, -1.0))), direction[None, :],
        )[0]
    return blender_module.LightData(
        light.pos_world, rotation, light.intensity, light_type,
        light.shadow_soft_size,
    )


def _image_paths(output_dir: Path) -> tuple[Path, ...]:
    """Return persisted Blender TIFFs in deterministic lexical order."""
    return tuple(sorted((output_dir / "images").glob("*.tiff")))


def _normalise_images(images: np.ndarray) -> np.ndarray:
    """Normalise legacy single-frame image arrays to the common layout."""
    if images.ndim == 2:
        return images[None, None, :, :, None]
    return images.transpose(2, 0, 1)[None, :, :, :, None]


def _normalise_deformed_images(images: np.ndarray, camera_count: int) -> np.ndarray:
    """Normalise legacy deformation stacks to ``frame, camera, y, x, channel``."""
    if images.ndim == 2:
        return images[None, None, :, :, None]
    frames = images.shape[2] // camera_count
    return images.reshape(
        images.shape[0], images.shape[1], frames, camera_count,
    ).transpose(2, 3, 0, 1)[:, :, :, :, None]


def blender_available() -> bool:
    """Return whether this interpreter can execute the Blender backend."""
    return _blender_unavailable_reason() is None


def blender_gpu_available() -> bool:
    """Return whether Blender reports a supported Cycles GPU device.

    Returns
    -------
    bool
        ``True`` when Blender's Cycles preferences report a CUDA, OptiX, HIP,
        Metal, or oneAPI device. ``False`` if Blender is unavailable.
    """
    if _blender_unavailable_reason() is not None:
        return False
    blender_module = importlib.import_module("pyvale.blender")
    return bool(blender_module.Tools.check_for_GPU())


def _blender_unavailable_reason() -> str | None:
    """Return an actionable reason when Blender cannot run here."""
    if sys.version_info[:2] != (3, 13):
        return "Blender requires Python 3.13 and the pyvale blender extra."
    try:
        importlib.import_module("pyvale.blender")
    except Exception as exception:
        return f"Blender is not available: {exception}"
    return None
