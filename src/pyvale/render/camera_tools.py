# ============================================================================
# pyvale: the python validation engine
# License: MIT
# Copyright (C) 2026 Sceptical Rabbit (Lloyd Fletcher)
# ============================================================================
"""Generic orthographic camera-grid operations for image warping."""

import numpy as np
from scipy.signal import convolve2d
from scipy.spatial.transform import Rotation

from .camera import Camera
from .camera_stereo import CameraStereo


class CameraTools:
    """Camera-grid helpers shared by planar image-warp renderers."""

    @staticmethod
    def pixel_vec_leng(field_of_view: np.ndarray,
                       leng_per_px: float) -> tuple[np.ndarray, np.ndarray]:
        """Build pixel-centre coordinate vectors for an orthographic camera.

        Parameters
        ----------
        field_of_view : numpy.ndarray
            Physical image extent in ``(width, height)`` order.
        leng_per_px : float
            Physical length represented by one pixel.

        Returns
        -------
        tuple[numpy.ndarray, numpy.ndarray]
            Pixel-centre coordinates along the horizontal and vertical axes.
        """
        return (
            np.arange(leng_per_px / 2.0, field_of_view[0], leng_per_px),
            np.arange(leng_per_px / 2.0, field_of_view[1], leng_per_px),
        )

    @staticmethod
    def pixel_grid_leng(field_of_view: np.ndarray,
                        leng_per_px: float) -> tuple[np.ndarray, np.ndarray]:
        """Build pixel-centre coordinate grids for an orthographic camera.

        Parameters
        ----------
        field_of_view : numpy.ndarray
            Physical image extent in ``(width, height)`` order.
        leng_per_px : float
            Physical length represented by one pixel.

        Returns
        -------
        tuple[numpy.ndarray, numpy.ndarray]
            Horizontal and vertical pixel-centre grids.
        """
        return np.meshgrid(*CameraTools.pixel_vec_leng(field_of_view, leng_per_px))

    @staticmethod
    def subpixel_vec_leng(field_of_view: np.ndarray,
                          leng_per_px: float,
                          subsample: int) -> tuple[np.ndarray, np.ndarray]:
        """Build sub-pixel-centre coordinate vectors.

        Parameters
        ----------
        field_of_view : numpy.ndarray
            Physical image extent in ``(width, height)`` order.
        leng_per_px : float
            Physical length represented by one output pixel.
        subsample : int
            Number of sub-pixels in each pixel direction.

        Returns
        -------
        tuple[numpy.ndarray, numpy.ndarray]
            Sub-pixel-centre coordinates along the two image axes.
        """
        spacing = leng_per_px / subsample
        return (
            np.arange(spacing / 2.0, field_of_view[0], spacing),
            np.arange(spacing / 2.0, field_of_view[1], spacing),
        )

    @staticmethod
    def subpixel_grid_leng(field_of_view: np.ndarray,
                           leng_per_px: float,
                           subsample: int) -> tuple[np.ndarray, np.ndarray]:
        """Build sub-pixel-centre coordinate grids.

        Parameters
        ----------
        field_of_view : numpy.ndarray
            Physical image extent in ``(width, height)`` order.
        leng_per_px : float
            Physical length represented by one output pixel.
        subsample : int
            Number of sub-pixels in each pixel direction.

        Returns
        -------
        tuple[numpy.ndarray, numpy.ndarray]
            Horizontal and vertical sub-pixel-centre grids.
        """
        return np.meshgrid(
            *CameraTools.subpixel_vec_leng(field_of_view, leng_per_px, subsample),
        )

    @staticmethod
    def crop_image_rectangle(image: np.ndarray,
                             pixels_count: np.ndarray) -> np.ndarray:
        """Crop an image to its camera extent from the upper-left corner.

        Parameters
        ----------
        image : numpy.ndarray
            Source image with rows followed by columns.
        pixels_count : numpy.ndarray
            Requested image size in ``(width, height)`` order.

        Returns
        -------
        numpy.ndarray
            View of the requested upper-left image rectangle.
        """
        return image[:pixels_count[1], :pixels_count[0]]

    @staticmethod
    def average_subpixel_image(image: np.ndarray, subsample: int) -> np.ndarray:
        """Average square sub-pixel blocks into output pixels.

        Parameters
        ----------
        image : numpy.ndarray
            Two-dimensional sub-pixel image.
        subsample : int
            Number of sub-pixels in each output-pixel direction.

        Returns
        -------
        numpy.ndarray
            Downsampled image. The input is returned unchanged for a factor of
            one or less.
        """
        if subsample <= 1:
            return image
        kernel = np.ones((subsample, subsample)) / (subsample ** 2)
        convolved = convolve2d(image, kernel, mode="same")
        start = round(subsample / 2.0) - 1
        return convolved[start::subsample, start::subsample]

    @staticmethod
    def blender_camera_from_resolution(
        pixels_num: np.ndarray,
        pixels_size: np.ndarray,
        working_dist: float,
        resolution: float,
    ) -> Camera:
        """Create a perspective camera from working distance and resolution."""
        focal_length = CameraTools.focal_length_from_resolution(
            pixels_size, working_dist, resolution,
        )
        return Camera(
            pixels_num, pixels_size, np.array((0.0, 0.0, working_dist)),
            Rotation.identity(), np.zeros(3), focal_length,
        )

    @staticmethod
    def focal_length_from_resolution(
        pixels_size: np.ndarray,
        working_dist: float,
        resolution: float,
    ) -> float:
        """Calculate the focal length needed for a Blender image resolution.

        Parameters
        ----------
        pixels_size : numpy.ndarray
            Pixel dimensions in world units.
        working_dist : float
            Camera-to-region working distance.
        resolution : float
            Requested world length per output pixel.

        Returns
        -------
        float
            Required focal length in world units.
        """
        return working_dist * pixels_size[0] / resolution

    @staticmethod
    def blender_field_of_view(camera: Camera) -> tuple[float, float]:
        """Calculate Blender's perspective field of view for a camera.

        Parameters
        ----------
        camera : Camera
            Perspective camera whose world distance and focal length are used.

        Returns
        -------
        tuple[float, float]
            Horizontal and vertical field of view in world units.
        """
        image_distance = np.linalg.norm(camera.pos_world - camera.roi_cent_world)
        field_x = (camera.pixels_num[0] * camera.pixels_size[0]
                   * image_distance / camera.focal_length)
        field_y = camera.pixels_num[1] / camera.pixels_num[0] * field_x
        return float(field_x), float(field_y)

    @staticmethod
    def blender_mm_per_pixel(camera: Camera) -> float:
        """Calculate the horizontal Blender image resolution in world units.

        Parameters
        ----------
        camera : Camera
            Perspective camera defining the Blender view.

        Returns
        -------
        float
            Horizontal world length represented by one pixel.
        """
        return CameraTools.blender_field_of_view(camera)[0] / camera.pixels_num[0]

    @staticmethod
    def faceon_stereo_cameras(
        camera: Camera,
        stereo_angle: float,
    ) -> CameraStereo:
        """Create face-on stereo cameras matching the legacy Blender helper."""
        camera_0 = camera
        baseline = camera.pos_world[2] * np.tan(np.radians(stereo_angle))
        camera_1 = Camera(
            camera.pixels_num.copy(), camera.pixels_size.copy(),
            camera.pos_world + np.array((baseline, 0.0, 0.0)),
            Rotation.from_euler(
                "xyz", (0.0, np.radians(stereo_angle), 0.0),
            ),
            camera.roi_cent_world.copy(), camera.focal_length, camera.sub_sample,
        )
        return CameraStereo(camera_0, camera_1)

    @staticmethod
    def symmetric_stereo_cameras(
        camera: Camera,
        stereo_angle: float,
    ) -> CameraStereo:
        """Create symmetric convergent stereo cameras from one reference view."""
        baseline = 2.0 * camera.pos_world[2] * np.tan(
            np.radians(stereo_angle) / 2.0,
        )
        common = (camera.pixels_num.copy(), camera.pixels_size.copy())
        camera_0 = Camera(
            *common,
            camera.pos_world - np.array((baseline / 2.0, 0.0, 0.0)),
            Rotation.from_euler("xyz", (0.0, -np.radians(stereo_angle / 2.0), 0.0)),
            camera.roi_cent_world.copy(), camera.focal_length, camera.sub_sample,
        )
        camera_1 = Camera(
            *common,
            camera.pos_world + np.array((baseline / 2.0, 0.0, 0.0)),
            Rotation.from_euler("xyz", (0.0, np.radians(stereo_angle / 2.0), 0.0)),
            camera.roi_cent_world.copy(), camera.focal_length, camera.sub_sample,
        )
        return CameraStereo(camera_0, camera_1)


__all__ = ["CameraTools"]
