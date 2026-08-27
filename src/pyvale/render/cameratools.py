"""Generic orthographic camera-grid and stereo-camera operations."""

import numpy as np
from scipy.signal import convolve2d
from scipy.spatial.transform import Rotation

from .camera import Camera
from .camerastereo import CameraStereo


def pixel_vec_leng(
    field_of_view: np.ndarray,
    pixels_size: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Build pixel-centre coordinate vectors for an orthographic camera."""
    return (
        np.arange(pixels_size / 2.0, field_of_view[0], pixels_size),
        np.arange(pixels_size / 2.0, field_of_view[1], pixels_size),
    )


def pixel_grid_leng(
    field_of_view: np.ndarray,
    pixels_size: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Build pixel-centre coordinate grids for an orthographic camera."""
    return np.meshgrid(*pixel_vec_leng(field_of_view, pixels_size))


def subpixel_vec_leng(
    field_of_view: np.ndarray,
    pixels_size: float,
    subsample: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Build sub-pixel-centre coordinate vectors."""
    spacing = pixels_size / subsample
    return (
        np.arange(spacing / 2.0, field_of_view[0], spacing),
        np.arange(spacing / 2.0, field_of_view[1], spacing),
    )


def subpixel_grid_leng(
    field_of_view: np.ndarray,
    pixels_size: float,
    subsample: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Build sub-pixel-centre coordinate grids."""
    return np.meshgrid(
        *subpixel_vec_leng(field_of_view, pixels_size, subsample),
    )


def crop_image_rectangle(
    image: np.ndarray,
    pixels_num: np.ndarray,
) -> np.ndarray:
    """Crop an image to its camera extent from the upper-left corner."""
    return image[: pixels_num[1], : pixels_num[0]]


def average_subpixel_image(image: np.ndarray, subsample: int) -> np.ndarray:
    """Average square sub-pixel blocks into output pixels."""
    if subsample <= 1:
        return image

    kernel = np.ones((subsample, subsample)) / (subsample**2)
    convolved = convolve2d(image, kernel, mode="same")
    start = round(subsample / 2.0) - 1
    return convolved[start::subsample, start::subsample]


def faceon_stereo_cameras(
    camera: Camera,
    stereo_angle: float,
) -> CameraStereo:
    """Create face-on stereo cameras from one reference view."""
    baseline = camera.pos_world[2] * np.tan(np.radians(stereo_angle))
    camera_1 = Camera(
        pixels_num=camera.pixels_num.copy(),
        pixels_size=camera.pixels_size.copy(),
        pos_world=camera.pos_world + np.array((baseline, 0.0, 0.0)),
        rot_world=Rotation.from_euler(
            "xyz",
            (0.0, np.radians(stereo_angle), 0.0),
        ),
        roi_cent_world=camera.roi_cent_world.copy(),
        focal_length=camera.focal_length,
        subsample=camera.subsample,
    )
    return CameraStereo(camera, camera_1)


def symmetric_stereo_cameras(
    camera: Camera,
    stereo_angle: float,
) -> CameraStereo:
    """Create symmetric convergent cameras from one reference view."""
    half_angle = stereo_angle / 2.0
    baseline = (
        2.0
        * camera.pos_world[2]
        * np.tan(
            np.radians(half_angle),
        )
    )

    def make_camera(offset: float, angle: float) -> Camera:
        return Camera(
            pixels_num=camera.pixels_num.copy(),
            pixels_size=camera.pixels_size.copy(),
            pos_world=camera.pos_world + np.array((offset, 0.0, 0.0)),
            rot_world=Rotation.from_euler(
                "xyz",
                (0.0, np.radians(angle), 0.0),
            ),
            roi_cent_world=camera.roi_cent_world.copy(),
            focal_length=camera.focal_length,
            subsample=camera.subsample,
        )

    return CameraStereo(
        make_camera(-baseline / 2.0, -half_angle),
        make_camera(baseline / 2.0, half_angle),
    )


__all__ = [
    "average_subpixel_image",
    "crop_image_rectangle",
    "faceon_stereo_cameras",
    "pixel_grid_leng",
    "pixel_vec_leng",
    "subpixel_grid_leng",
    "subpixel_vec_leng",
    "symmetric_stereo_cameras",
]
