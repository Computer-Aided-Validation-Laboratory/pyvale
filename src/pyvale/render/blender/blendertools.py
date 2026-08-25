# ============================================================================
# pyvale: the python validation engine
# License: MIT
# Copyright (C) 2026 Sceptical Rabbit (Lloyd Fletcher)
# ============================================================================
"""Camera helpers specific to the Blender rendering backend."""

import numpy as np
from scipy.spatial.transform import Rotation

from ..camera import Camera


class BlenderTools:
    """Utilities for constructing and measuring Blender cameras."""

    @staticmethod
    def blender_camera_from_resolution(
        pixels_num: np.ndarray,
        pixels_size: np.ndarray,
        working_dist: float,
        resolution: float,
    ) -> Camera:
        """Create a perspective camera from working distance and resolution."""
        focal_length = BlenderTools.focal_length_from_resolution(
            pixels_size,
            working_dist,
            resolution,
        )
        return Camera(
            pixels_num,
            pixels_size,
            np.array((0.0, 0.0, working_dist)),
            Rotation.identity(),
            np.zeros(3),
            focal_length,
        )

    @staticmethod
    def focal_length_from_resolution(
        pixels_size: np.ndarray,
        working_dist: float,
        resolution: float,
    ) -> float:
        """Calculate the focal length for a requested image resolution."""
        return working_dist * pixels_size[0] / resolution

    @staticmethod
    def blender_field_of_view(camera: Camera) -> tuple[float, float]:
        """Calculate Blender's perspective field of view for a camera."""
        image_distance = np.linalg.norm(
            camera.pos_world - camera.roi_cent_world,
        )
        field_x = (
            camera.pixels_num[0]
            * camera.pixels_size[0]
            * image_distance
            / camera.focal_length
        )
        field_y = camera.pixels_num[1] / camera.pixels_num[0] * field_x
        return float(field_x), float(field_y)

    @staticmethod
    def blender_mm_per_pixel(camera: Camera) -> float:
        """Calculate the horizontal world length represented by one pixel."""
        return (
            BlenderTools.blender_field_of_view(camera)[0]
            / camera.pixels_num[0]
        )


__all__ = ["BlenderTools"]
