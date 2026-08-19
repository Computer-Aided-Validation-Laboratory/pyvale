# ============================================================================
# pyvale: the python validation engine
# License: MIT
# Copyright (C) 2026 Sceptical Rabbit (Lloyd Fletcher)
# ============================================================================
"""Blender implementation of the unified 3D renderer interface."""

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
    """Stable Blender controls accepted by the unified adapter.

    Parameters
    ----------
    output_dir : pathlib.Path
        Directory used for Blender output files.
    samples : int, optional
        Number of Blender render samples.
    threads : int, optional
        Number of Blender worker threads.
    """

    output_dir: Path
    samples: int = 2
    threads: int = 1


@dataclass(frozen=True, slots=True)
class _BlenderPlan:
    """Validated Blender scene data ready for scene construction.

    Parameters
    ----------
    meshes : tuple[Mesh, ...]
        Validated render meshes.
    cameras : tuple[Camera, ...]
        Validated render cameras.
    lights : tuple[Light, ...]
        Validated lights requested for the scene.
    """
    meshes: tuple[Mesh, ...]
    cameras: tuple[Camera, ...]
    lights: tuple[Light, ...]


class Blender(IRenderer3D):
    """Render common meshes and cameras through Blender's existing scene API.

    Parameters
    ----------
    config : BlenderConfig
        Output and execution controls for the Blender adapter.
    """

    def __init__(self, config: BlenderConfig) -> None:
        """Store the configuration used by subsequent render requests.

        Parameters
        ----------
        config : BlenderConfig
            Output and execution controls for the adapter.
        """
        self.config = config

    def verify_input(
        self,
        meshes: Sequence[Mesh],
        cameras: Sequence[Camera],
        lights: Sequence[Light] | None = None,
    ) -> _BlenderPlan:
        """Verify a scene before Blender resets or creates a scene.

        Parameters
        ----------
        meshes : Sequence[Mesh]
            Meshes passed to Blender's scene API.
        cameras : Sequence[Camera]
            Cameras passed to Blender's scene API.
        lights : Sequence[Light] or None, optional
            Lights requested for the scene.

        Returns
        -------
        _BlenderPlan
            Validated scene data ready for Blender scene construction.

        Raises
        ------
        RenderInputError
            If the scene or Blender configuration is invalid.
        """
        issues = list(verify_scene_3d(meshes, cameras, lights))
        if self.config.samples <= 0 or self.config.threads <= 0:
            issues.append(ValidationIssue("config", "VALUE", "Samples and threads must be positive."))
        raise_if_issues(tuple(issues))
        return _BlenderPlan(tuple(meshes), tuple(cameras), tuple(lights or ()))

    def _render(self, render_plan: object) -> RenderResult:
        """Build a minimal Blender scene and normalise image output.

        Parameters
        ----------
        render_plan : object
            Plan returned by :meth:`verify_input`.

        Returns
        -------
        RenderResult
            Blender images in the common render-result layout.

        Raises
        ------
        TypeError
            If ``render_plan`` was not created by this renderer.
        """
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
