# ==============================================================================
# pyvale: the python validation engine
# License: MIT
# Copyright (C) 2026 Sceptical Rabbit (Lloyd Fletcher)
# ==============================================================================
"""Camera placement, orientation, projection, and stereo configuration
helpers.
"""

from collections.abc import Sequence
from dataclasses import replace

import numpy as np
import riley
from scipy.signal import convolve2d
from scipy.spatial.transform import Rotation

from .camera import Camera
from .camerastereo import CameraStereo
from .mesh import Mesh3D


def cam_look_at(
    camera: Camera,
    target: Sequence[float],
    up: Sequence[float] = (0.0, 1.0, 0.0),
) -> Camera:
    """Orient a camera so its optical axis points towards a target location.

    Parameters
    ----------
    camera : Camera
        Camera to reorient.
    target : Sequence[float]
        3D world coordinates the camera should aim at.
    up : Sequence[float], optional
        Preferred upward world direction (default is +Y).

    Returns
    -------
    Camera
        A copy of the camera with updated rotation and ROI center.
    """
    target_vec = np.asarray(target, dtype=np.float64)
    pos_vec = np.asarray(camera.pos_world, dtype=np.float64)
    view_dir = target_vec - pos_vec
    dist = np.linalg.norm(view_dir)

    if dist < 1.0e-12:
        raise ValueError("Camera position and look-at target are coincident.")

    forward = view_dir / dist
    z_cam = -forward

    up_vec = np.asarray(up, dtype=np.float64)
    up_norm = np.linalg.norm(up_vec)
    if up_norm < 1.0e-12:
        up_vec = np.array((0.0, 1.0, 0.0))
    else:
        up_vec = up_vec / up_norm

    x_cam = np.cross(up_vec, z_cam)
    x_norm = np.linalg.norm(x_cam)
    if x_norm < 1.0e-6:
        fallback_up = np.array((0.0, 0.0, 1.0))
        if abs(np.dot(fallback_up, z_cam)) > 0.9:
            fallback_up = np.array((1.0, 0.0, 0.0))
        x_cam = np.cross(fallback_up, z_cam)
        x_norm = np.linalg.norm(x_cam)

    x_cam = x_cam / x_norm
    y_cam = np.cross(z_cam, x_cam)
    y_cam = y_cam / np.linalg.norm(y_cam)

    rot_matrix = np.column_stack((x_cam, y_cam, z_cam))
    rot_world = Rotation.from_matrix(rot_matrix)

    return replace(
        camera,
        rot_world=rot_world,
        roi_cent_world=target_vec.copy(),
    )


def cam_frame_points(
    camera: Camera,
    points: np.ndarray,
    fill: float = 1.0,
) -> Camera:
    """Move a camera along its view direction to frame a set of points.

    Parameters
    ----------
    camera : Camera
        Camera to position.
    points : numpy.ndarray
        Array of 3D point coordinates to fit inside the sensor.
    fill : float, optional
        Target fraction of the field of view to fill (default is 1.0).

    Returns
    -------
    Camera
        A copy of the camera positioned to frame the points.
    """
    pts = np.asarray(points, dtype=np.float64)
    if pts.size == 0:
        raise ValueError("Cannot frame an empty set of points.")

    rot_euler = tuple(camera.rot_world.as_euler("xyz"))
    pos = riley.pos_fill_frame_from_rot(
        pts,
        tuple(camera.pixels_num),
        tuple(camera.pixels_size),
        camera.focal_length,
        rot_euler,
        fill,
    )
    roi = riley.roi_cent_from_coords(pts)

    return replace(
        camera,
        pos_world=np.asarray(pos, dtype=np.float64),
        roi_cent_world=np.asarray(roi, dtype=np.float64),
    )


def cam_frame_mesh(
    camera: Camera,
    mesh: Mesh3D,
    fill: float = 1.0,
) -> Camera:
    """Position a camera along its view direction to frame a mesh."""
    return cam_frame_points(camera, mesh.coords, fill=fill)


def cam_frame_scene(
    camera: Camera,
    meshes: Sequence[Mesh3D],
    fill: float = 1.0,
) -> Camera:
    """Position a camera along its view direction to frame all meshes."""
    valid_coords = [m.coords for m in meshes if len(m.coords) > 0]
    if not valid_coords:
        raise ValueError("Cannot frame a scene with no mesh coordinates.")
    all_pts = np.concatenate(valid_coords, axis=0)
    return cam_frame_points(camera, all_pts, fill=fill)


def cam_project_points(
    camera: Camera,
    points: np.ndarray,
) -> np.ndarray:
    """Project 3D world points to 2D image pixel coordinates.

    Parameters
    ----------
    camera : Camera
        Perspective camera model.
    points : numpy.ndarray
        Array of shape ``(N, 3)`` in world coordinates.

    Returns
    -------
    numpy.ndarray
        Projected image coordinates of shape ``(N, 2)`` in pixel units.
    """
    pts = np.asarray(points, dtype=np.float64)
    if pts.ndim == 1:
        pts = pts[None, :]

    pos = np.asarray(camera.pos_world, dtype=np.float64)
    rel_pts = pts - pos

    rot_mat = camera.rot_world.as_matrix()
    cam_pts = rel_pts @ rot_mat

    depth = -cam_pts[:, 2]
    with np.errstate(divide="ignore", invalid="ignore"):
        proj_x = camera.focal_length * (cam_pts[:, 0] / depth)
        proj_y = camera.focal_length * (cam_pts[:, 1] / depth)

    cx = 0.5 * camera.pixels_num[0]
    cy = 0.5 * camera.pixels_num[1]

    px_u = cx + proj_x / camera.pixels_size[0]
    px_v = cy + proj_y / camera.pixels_size[1]

    return np.column_stack((px_u, px_v))


def cam_stereo_faceon(
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


def cam_stereo_symmetric(
    camera: Camera,
    stereo_angle: float,
) -> CameraStereo:
    """Create symmetric convergent cameras from one reference view."""
    half_angle = stereo_angle / 2.0
    baseline = 2.0 * camera.pos_world[2] * np.tan(np.radians(half_angle))

    def make_cam(offset: float, angle: float) -> Camera:
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
        make_cam(-baseline / 2.0, -half_angle),
        make_cam(baseline / 2.0, half_angle),
    )


# Backwards compatibility aliases
faceon_stereo_cameras = cam_stereo_faceon
symmetric_stereo_cameras = cam_stereo_symmetric


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


__all__ = [
    "average_subpixel_image",
    "cam_frame_mesh",
    "cam_frame_points",
    "cam_frame_scene",
    "cam_look_at",
    "cam_project_points",
    "cam_stereo_faceon",
    "cam_stereo_symmetric",
    "crop_image_rectangle",
    "faceon_stereo_cameras",
    "pixel_grid_leng",
    "pixel_vec_leng",
    "subpixel_grid_leng",
    "subpixel_vec_leng",
    "symmetric_stereo_cameras",
]
