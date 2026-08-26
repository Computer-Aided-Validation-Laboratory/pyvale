"""Blender implementation of pyvale's unified renderer API."""

from .adapter import Blender, blender_available, blender_gpu_available
from .blendertools import (
    blender_camera_from_resolution,
    blender_field_of_view,
    blender_mm_per_pixel,
    focal_length_from_resolution,
)
from .calibration import (
    BlenderCalibrationData,
    BlenderCalibrationTarget,
    calibration_image_count,
    render_calibration_images,
)
from .config import BlenderConfig, EBlenderDevice, EBlenderEngine
from .shader import BlenderImageShader, BlenderMaterial, BlenderTextureShader

__all__ = [
    "Blender",
    "BlenderCalibrationData",
    "BlenderCalibrationTarget",
    "BlenderConfig",
    "BlenderImageShader",
    "BlenderMaterial",
    "BlenderTextureShader",
    "EBlenderDevice",
    "EBlenderEngine",
    "blender_available",
    "blender_camera_from_resolution",
    "blender_field_of_view",
    "blender_gpu_available",
    "blender_mm_per_pixel",
    "calibration_image_count",
    "focal_length_from_resolution",
    "render_calibration_images",
]
