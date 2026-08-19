# ============================================================================
# pyvale: the python validation engine
# License: MIT
# Copyright (C) 2026 Sceptical Rabbit (Lloyd Fletcher)
# ============================================================================
"""Blender implementation of the unified 3D renderer lifecycle."""

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .camera import Camera
from .errors import ValidationIssue
from .light import Light
from .mesh import Mesh
from .renderer3d import IRenderer3D
from .result import RenderResult
from .verifyinput import raise_if_issues, verify_scene_3d


@dataclass(frozen=True, slots=True)
class BlenderConfig:
    """Small set of stable Blender controls for the unified adapter."""

    output_dir: Path
    samples: int = 2
    threads: int = 1


@dataclass(frozen=True, slots=True)
class _BlenderPlan:
    meshes: tuple[Mesh, ...]
    cameras: tuple[Camera, ...]
    lights: tuple[Light, ...]


class Blender(IRenderer3D):
    """Render common meshes and cameras through Blender's existing scene API."""

    def __init__(self, config: BlenderConfig) -> None:
        self.config = config

    def verify_input(
        self,
        meshes: Sequence[Mesh],
        cameras: Sequence[Camera],
        lights: Sequence[Light] | None = None,
    ) -> _BlenderPlan:
        """Verify the scene before Blender resets or creates a scene."""
        issues = list(verify_scene_3d(meshes, cameras, lights))
        if self.config.samples <= 0 or self.config.threads <= 0:
            issues.append(ValidationIssue("config", "VALUE", "Samples and threads must be positive."))
        raise_if_issues(tuple(issues))
        return _BlenderPlan(tuple(meshes), tuple(cameras), tuple(lights or ()))

    def _render(self, render_plan: object) -> RenderResult:
        """Build a minimal Blender scene and normalise image output."""
        if not isinstance(render_plan, _BlenderPlan):
            raise TypeError("Blender received an invalid render plan.")
        from pyvale.blender import RenderData, Scene

        scene = Scene()
        for mesh in render_plan.meshes:
            scene.add_part(mesh, 3)
        for camera in render_plan.cameras:
            scene.add_camera(camera)
        self.config.output_dir.mkdir(parents=True, exist_ok=True)
        render_data = RenderData(
            cam_data=(render_plan.cameras[0] if len(render_plan.cameras) == 1
                      else (render_plan.cameras[0], render_plan.cameras[1])),
            base_dir=self.config.output_dir,
            samples=self.config.samples,
            threads=self.config.threads,
        )
        image = scene.render_single_image(render_data, stage_image=True)
        if image is None:
            return RenderResult(images=None)
        images = np.asarray(image)
        if images.ndim == 2:
            images = images[None, None, :, :, None]
        elif images.ndim == 3:
            images = images.transpose(2, 0, 1)[None, :, :, :, None]
        return RenderResult(images=images)


__all__ = ["Blender", "BlenderConfig"]
