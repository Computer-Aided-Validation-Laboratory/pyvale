# ============================================================================
# pyvale: the python validation engine
# License: MIT
# Copyright (C) 2026 Sceptical Rabbit (Lloyd Fletcher)
# ============================================================================
"""Blender calibration-target configuration helpers."""

from collections.abc import Sequence
from dataclasses import dataclass
import importlib
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from ..camera import Camera
from ..camerastereo import CameraStereo
from ..light import Light
from ..result import RenderResult
from .adapter import _blender_unavailable_reason, _legacy_light
from .config import BlenderConfig
from .shader import BlenderMaterial


@dataclass(frozen=True, slots=True)
class BlenderCalibrationData:
    """Describe the pose sweep used to render a camera calibration target.

    Parameters
    ----------
    angle_lims : tuple[float, float], optional
        Inclusive lower and upper target-angle limits in degrees.
    angle_step : float, optional
        Angular increment in degrees.
    plunge_lims : tuple[float, float], optional
        Inclusive lower and upper target-depth limits.
    plunge_step : float, optional
        Target-depth increment.
    x_limit, y_limit : float or None, optional
        Optional lateral target-position limits retained for calibration scene
        generation.
    max_images : int or None, optional
        Maximum number of TIFF files to render. ``None`` renders every pose.
    """

    angle_lims: tuple[float, float] = (-10.0, 10.0)
    angle_step: float = 5.0
    plunge_lims: tuple[float, float] = (-5.0, 5.0)
    plunge_step: float = 5.0
    x_limit: float | None = None
    y_limit: float | None = None
    max_images: int | None = None


@dataclass(frozen=True, slots=True)
class BlenderCalibrationTarget:
    """A textured planar target used to generate calibration images.

    Parameters
    ----------
    size : numpy.ndarray
        Target width, height, and thickness in world units.
    image_path : pathlib.Path
        Calibration-target texture image.
    millimetres_per_pixel : float
        Texture resolution used by Blender UV unwrapping.
    material : BlenderMaterial, optional
        Backend-owned material controls for the target.
    """

    size: np.ndarray
    image_path: Path
    millimetres_per_pixel: float
    material: BlenderMaterial = BlenderMaterial()

    def __post_init__(self) -> None:
        """Normalise target array and texture path inputs."""
        object.__setattr__(self, "size", np.asarray(self.size, dtype=np.float64))
        object.__setattr__(self, "image_path", Path(self.image_path))


def calibration_image_count(data: BlenderCalibrationData) -> int:
    """Return the number of legacy Blender calibration-target images.

    Parameters
    ----------
    data : BlenderCalibrationData
        Calibration target pose-sweep configuration.

    Returns
    -------
    int
        Number of target poses in the historical nine-position lateral sweep.
    """
    plunge_steps = (data.plunge_lims[1] - data.plunge_lims[0]) / data.plunge_step
    angle_steps = (data.angle_lims[1] - data.angle_lims[0]) / data.angle_step
    return int((plunge_steps + 1.0) * (angle_steps + 1.0) ** 2 * 9)


def render_calibration_images(
    target: BlenderCalibrationTarget,
    cameras: CameraStereo | Sequence[Camera],
    config: BlenderConfig,
    data: BlenderCalibrationData = BlenderCalibrationData(),
    lights: Sequence[Light] | None = None,
) -> RenderResult:
    """Render a legacy-compatible Blender calibration-target pose sweep.

    Parameters
    ----------
    target : BlenderCalibrationTarget
        Textured physical calibration target.
    cameras : CameraStereo or Sequence[Camera]
        Exactly two perspective cameras defining the calibration rig.
    config : BlenderConfig
        Blender engine, image output, and sampling controls.
    data : BlenderCalibrationData, optional
        Target translation and rotation sweep controls.
    lights : Sequence[Light] or None, optional
        Optional scene lights. ``None`` preserves Blender's dark default world.

    Returns
    -------
    RenderResult
        File-only result containing the calibration TIFF paths.

    Raises
    ------
    RenderInputError
        If input is invalid or the Blender backend is unavailable.
    """
    reason = _blender_unavailable_reason()
    if reason is not None:
        raise RuntimeError(reason)
    if not isinstance(target, BlenderCalibrationTarget):
        raise TypeError("target must be a BlenderCalibrationTarget.")
    if not isinstance(config, BlenderConfig):
        raise TypeError("config must be a BlenderConfig.")
    if not isinstance(data, BlenderCalibrationData):
        raise TypeError("data must be a BlenderCalibrationData.")
    if data.max_images is not None and data.max_images < 1:
        raise ValueError("data.max_images must be positive when specified.")
    if target.size.shape != (3,) or np.any(target.size <= 0.0):
        raise ValueError("target.size must contain three positive dimensions.")
    if isinstance(cameras, CameraStereo):
        camera_data = (cameras.cam_data_0, cameras.cam_data_1)
    else:
        camera_data = tuple(cameras)
    if len(camera_data) != 2 or not all(isinstance(camera, Camera)
                                         for camera in camera_data):
        raise ValueError("Calibration rendering requires exactly two Cameras.")

    blender = importlib.import_module("pyvale.blender")
    scene = blender.Scene()
    target_object = scene.add_cal_target(target.size)
    for camera in camera_data:
        scene.add_camera(camera)
    for light in lights or ():
        scene.add_light(_legacy_light(blender, light))
    scene.add_speckle(
        target_object, target.image_path,
        blender.MaterialData(
            roughness=target.material.roughness,
            metallic=target.material.metallic,
            interpolant=target.material.interpolant,
        ),
        target.millimetres_per_pixel,
        cal=True,
    )
    config.output_dir.mkdir(parents=True, exist_ok=True)
    render_data = blender.RenderData(
        cam_data=tuple(_legacy_calibration_camera(camera) for camera in camera_data),
        base_dir=config.output_dir,
        samples=config.samples,
        max_bounces=config.max_bounces,
        threads=config.threads,
        engine=blender.RenderEngine(config.engine.value),
    )
    legacy_data = blender.CalibrationData(
        angle_lims=data.angle_lims,
        angle_step=data.angle_step,
        plunge_lims=data.plunge_lims,
        plunge_step=data.plunge_step,
        x_limit=data.x_limit,
        y_limit=data.y_limit,
        max_images=data.max_images,
    )
    blender.Tools.render_calibration_images(render_data, legacy_data, target_object)
    paths = tuple(sorted((config.output_dir / "calimages").glob("*.tiff")))
    return RenderResult(None, paths)


def _legacy_calibration_camera(camera: Camera) -> SimpleNamespace:
    """Supply the derived camera data required by the legacy target loop."""
    values = {
        field: getattr(camera, field)
        for field in camera.__dataclass_fields__
    }
    values["image_dist"] = float(np.linalg.norm(
        camera.pos_world - camera.roi_cent_world,
    ))
    return SimpleNamespace(**values)


__all__ = [
    "BlenderCalibrationData",
    "BlenderCalibrationTarget",
    "calibration_image_count",
    "render_calibration_images",
]
