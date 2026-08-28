# ==============================================================================
# pyvale: the python validation engine
# License: MIT
# Copyright (C) 2026 Sceptical Rabbit (Lloyd Fletcher)
# ==============================================================================
"""Camera placement, orientation, projection, and stereo configuration
helpers.
"""

from collections.abc import Sequence
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np
import riley
import yaml
from scipy.signal import convolve2d
from scipy.spatial.transform import Rotation

from .camera import Camera
from .mesh import Mesh3D


StereoCameras = tuple[Camera, Camera]
"""The ordered pair of cameras that defines a stereo rig."""


@dataclass(frozen=True, slots=True)
class StereoExtrinsics:
    """Transform from camera-zero coordinates to camera-one coordinates.

    The relation is ``point_cam1 = rotation_cam1_from_cam0.apply(point_cam0)
    + translation_cam1_in_cam0``.  The camera rotations stored by
    :class:`Camera` transform camera coordinates into world coordinates.
    """

    rotation_cam1_from_cam0: Rotation
    translation_cam1_in_cam0: np.ndarray


@dataclass(frozen=True, slots=True)
class StereoAngles:
    """Relative orientation and optical-axis convergence of a stereo pair."""

    relative_euler_xyz_degrees: np.ndarray
    convergence_degrees: float


def cam_look_at(
    camera: Camera,
    target: np.ndarray,
    up: np.ndarray = np.array((0.0, 1.0, 0.0)),
) -> Camera:
    """Orient a camera so its optical axis points towards a target location.

    Parameters
    ----------
    camera : Camera
        Camera to reorient.
    target : numpy.ndarray
        3D world coordinates the camera should aim at.
    up : numpy.ndarray, optional
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


def cam_pos_fill_frame(
    points: np.ndarray,
    pixels_num: tuple[int, int] | np.ndarray,
    pixels_size: tuple[float, float] | np.ndarray,
    focal_length: float,
    rot_world: tuple[float, float, float] | np.ndarray | Rotation = (
        0.0,
        0.0,
        0.0,
    ),
    fill: float = 1.0,
) -> np.ndarray:
    """Calculate the world camera position to fill a fraction of the sensor.

    Parameters
    ----------
    points : numpy.ndarray
        Array of 3D point coordinates (e.g. mesh.coords) to frame.
    pixels_num : tuple[int, int] or numpy.ndarray
        Camera sensor resolution in pixels (num_x, num_y).
    pixels_size : tuple[float, float] or numpy.ndarray
        Pixel physical dimensions in metres.
    focal_length : float
        Camera focal length in metres.
    rot_world : tuple, numpy.ndarray, or Rotation, optional
        Euler angles in radians (xyz) or a scipy Rotation (default (0, 0, 0)).
    fill : float, optional
        Target fraction of the sensor field of view to fill (default 1.0).

    Returns
    -------
    numpy.ndarray
        Camera world coordinates [x, y, z] to frame the points.
    """
    pts = np.asarray(points, dtype=np.float64)
    if pts.ndim != 2 or pts.shape[1] != 3:
        raise ValueError("points must have shape (N, 3).")
    if pts.shape[0] == 0:
        raise ValueError("Cannot frame an empty set of points.")

    if fill <= 0.0:
        raise ValueError("fill must be positive.")

    pixel_counts = np.asarray(pixels_num, dtype=np.float64)
    pixel_sizes = np.asarray(pixels_size, dtype=np.float64)
    if (
        pixel_counts.shape != (2,)
        or pixel_sizes.shape != (2,)
        or np.any(pixel_counts <= 0.0)
        or np.any(pixel_sizes <= 0.0)
        or focal_length <= 0.0
    ):
        raise ValueError("Camera dimensions and focal length must be positive.")

    if isinstance(rot_world, Rotation):
        rotation = rot_world
    else:
        rotation = Rotation.from_euler(
            "xyz",
            np.asarray(rot_world, dtype=np.float64),
        )

    target = 0.5 * (np.min(pts, axis=0) + np.max(pts, axis=0))
    camera_points = rotation.inv().apply(pts - target)
    sensor_half_size = 0.5 * pixel_counts * pixel_sizes * fill

    distance_x = (
        camera_points[:, 2]
        + focal_length * np.abs(camera_points[:, 0]) / sensor_half_size[0]
    )
    distance_y = (
        camera_points[:, 2]
        + focal_length * np.abs(camera_points[:, 1]) / sensor_half_size[1]
    )
    distance = float(max(np.max(distance_x), np.max(distance_y)))
    clearance = np.finfo(np.float64).eps * max(1.0, abs(distance))
    distance = max(distance, float(np.max(camera_points[:, 2]))) + clearance

    return target + rotation.apply(np.array((0.0, 0.0, distance)))


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

    pos = cam_pos_fill_frame(
        pts,
        camera.pixels_num,
        camera.pixels_size,
        camera.focal_length,
        camera.rot_world,
        fill,
    )
    roi = riley.roi_cent_from_coords(pts)

    return replace(
        camera,
        pos_world=pos,
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

    valid_coords = [mesh.coords for mesh in meshes if len(mesh.coords) > 0]
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
    proj_x = camera.focal_length * (cam_pts[:, 0] / depth)
    proj_y = camera.focal_length * (cam_pts[:, 1] / depth)

    cx = 0.5 * camera.pixels_num[0]
    cy = 0.5 * camera.pixels_num[1]

    px_u = cx + proj_x / camera.pixels_size[0]
    px_v = cy + proj_y / camera.pixels_size[1]

    return np.column_stack((px_u, px_v))


def stereo_build_faceon(
    camera: Camera,
    convergence_degrees: float,
    roi_pos: np.ndarray | None = None,
) -> StereoCameras:
    """Build a pair with camera zero face-on and camera one converging.

    Camera zero is retained unchanged. Camera one is translated along camera
    zero's local positive X axis and aimed at ``roi_pos``. When no ROI is
    supplied, ``camera.roi_cent_world`` is used.
    """

    target = _stereo_target(camera, roi_pos)
    stand_off = np.linalg.norm(camera.pos_world - target)
    baseline = stand_off * np.tan(np.radians(convergence_degrees))
    baseline_dir = camera.rot_world.apply(np.array((1.0, 0.0, 0.0)))

    camera_1 = replace(
        camera,
        pos_world=camera.pos_world + baseline * baseline_dir,
        roi_cent_world=target,
    )

    return camera, cam_look_at(camera_1, target)


def stereo_build_symmetric(
    camera: Camera,
    convergence_degrees: float,
    roi_pos: np.ndarray | None = None,
) -> StereoCameras:
    """Build a symmetric convergent pair centred on a reference camera.

    The reference camera supplies the midpoint pose and intrinsics. Both
    returned cameras are placed on its local X axis and aimed at ``roi_pos``.
    """

    target = _stereo_target(camera, roi_pos)
    stand_off = np.linalg.norm(camera.pos_world - target)
    half_angle = 0.5 * np.radians(convergence_degrees)
    baseline = 2.0 * stand_off * np.tan(half_angle)
    baseline_dir = camera.rot_world.apply(np.array((1.0, 0.0, 0.0)))

    camera_0 = replace(
        camera,
        pos_world=camera.pos_world - 0.5 * baseline * baseline_dir,
        roi_cent_world=target,
    )

    camera_1 = replace(
        camera,
        pos_world=camera.pos_world + 0.5 * baseline * baseline_dir,
        roi_cent_world=target,
    )

    return cam_look_at(camera_0, target), cam_look_at(camera_1, target)


def stereo_calc_extrinsics(
    camera_0: Camera,
    camera_1: Camera,
) -> StereoExtrinsics:
    """Calculate the camera-zero to camera-one rigid transformation."""

    rotation = camera_1.rot_world.inv() * camera_0.rot_world
    translation = camera_1.rot_world.inv().apply(
        camera_0.pos_world - camera_1.pos_world
    )

    return StereoExtrinsics(rotation, translation)


def stereo_calc_baseline(camera_0: Camera, camera_1: Camera) -> float:
    """Calculate the Euclidean distance between the camera centres."""
    return float(np.linalg.norm(camera_1.pos_world - camera_0.pos_world))


def stereo_calc_stand_off(
    camera_0: Camera,
    camera_1: Camera,
    roi_pos: np.ndarray,
) -> float:
    """Calculate midpoint-to-ROI stand-off distance for a stereo pair."""

    midpoint = 0.5 * (camera_0.pos_world + camera_1.pos_world)
    roi = np.asarray(roi_pos, dtype=np.float64)

    return float(np.linalg.norm(midpoint - roi))


def stereo_calc_angles(camera_0: Camera, camera_1: Camera) -> StereoAngles:
    """Calculate relative Euler angles and optical-axis convergence."""

    extrinsics = stereo_calc_extrinsics(camera_0, camera_1)

    optical_0 = -camera_0.rot_world.as_matrix()[:, 2]
    optical_1 = -camera_1.rot_world.as_matrix()[:, 2]
    cosine = np.clip(np.dot(optical_0, optical_1), -1.0, 1.0)

    return StereoAngles(
        extrinsics.rotation_cam1_from_cam0.as_euler("xyz", degrees=True),
        float(np.degrees(np.arccos(cosine))),
    )


def stereo_build_from_calibration(
    calibration_path: Path,
    pos_world_0: np.ndarray,
    rot_world_0: Rotation,
    focal_length: float,
) -> StereoCameras:
    """Build stereo cameras from a legacy PyVale YAML calibration file."""

    parameters = yaml.safe_load(Path(calibration_path).read_text())

    camera_0 = _camera_from_calibration(
        parameters,
        0,
        pos_world_0,
        rot_world_0,
        focal_length,
    )

    rotation = Rotation.from_euler(
        "xyz",
        (
            parameters["Theta [deg]"],
            parameters["Phi [deg]"],
            parameters["Psi [deg]"],
        ),
        degrees=True,
    )

    translation = np.array(
        (
            parameters["Tx [mm]"],
            parameters["Ty [mm]"],
            parameters["Tz [mm]"],
        ),
        dtype=np.float64,
    )

    rot_world_1 = rot_world_0 * rotation.inv()
    pos_world_1 = np.asarray(pos_world_0, dtype=np.float64) - rot_world_1.apply(
        translation
    )

    camera_1 = _camera_from_calibration(
        parameters,
        1,
        pos_world_1,
        rot_world_1,
        focal_length,
    )
    
    return camera_0, camera_1


def stereo_save_calibration_yaml(
    camera_0: Camera,
    camera_1: Camera,
    calibration_path: Path,
) -> None:
    """Save two cameras in PyVale's legacy YAML calibration format."""
    Path(calibration_path).parent.mkdir(parents=True, exist_ok=True)
    Path(calibration_path).write_text(
        yaml.safe_dump(_stereo_calibration_parameters(camera_0, camera_1))
    )


def stereo_save_calibration_matchid(
    camera_0: Camera,
    camera_1: Camera,
    calibration_path: Path,
) -> None:
    """Save two cameras in the legacy MatchID ``.caldat`` format."""
    parameters = _stereo_calibration_parameters(camera_0, camera_1)
    Path(calibration_path).parent.mkdir(parents=True, exist_ok=True)
    Path(calibration_path).write_text(
        "\n".join(f"{key};{value}" for key, value in parameters.items())
    )


def _stereo_target(
    camera: Camera,
    roi_pos: np.ndarray | None,
) -> np.ndarray:
    """Return a validated stereo convergence target."""

    target = np.asarray(
        camera.roi_cent_world if roi_pos is None else roi_pos,
        dtype=np.float64,
    )

    if target.shape != (3,):
        raise ValueError("roi_pos must contain exactly three coordinates.")

    if np.linalg.norm(camera.pos_world - target) < 1.0e-12:
        raise ValueError("Camera position and stereo ROI are coincident.")

    return target


def _camera_from_calibration(
    parameters: dict[str, float],
    camera_index: int,
    pos_world: np.ndarray,
    rot_world: Rotation,
    focal_length: float,
) -> Camera:
    """Build one render camera from legacy calibration parameters."""

    prefix = f"Cam{camera_index}"

    pixels_num = np.array(
        (
            int(2.0 * parameters[f"{prefix}_Cx [pixels]"]),
            int(2.0 * parameters[f"{prefix}_Cy [pixels]"]),
        ),
    )

    pixels_size = np.array(
        (
            focal_length / parameters[f"{prefix}_Fx [pixels]"],
            focal_length / parameters[f"{prefix}_Fy [pixels]"],
        ),
    )

    return Camera(
        pixels_num=pixels_num,
        pixels_size=pixels_size,
        pos_world=pos_world,
        rot_world=rot_world,
        roi_cent_world=np.zeros(3),
        focal_length=focal_length,
        distortion_k1=parameters[f"{prefix}_Kappa 1"],
        distortion_k2=parameters[f"{prefix}_Kappa 2"],
        distortion_k3=parameters[f"{prefix}_Kappa 3"],
        distortion_p1=parameters[f"{prefix}_P1"],
        distortion_p2=parameters[f"{prefix}_P2"],
        c0=parameters[f"{prefix}_Cx [pixels]"],
        c1=parameters[f"{prefix}_Cy [pixels]"],
    )


def _stereo_calibration_parameters(
    camera_0: Camera,
    camera_1: Camera,
) -> dict[str, float]:
    """Format camera intrinsics and extrinsics for legacy serializers."""

    extrinsics = stereo_calc_extrinsics(camera_0, camera_1)
    rotation = extrinsics.rotation_cam1_from_cam0.as_euler("xyz", degrees=True)
    parameters: dict[str, float] = {}

    for camera_index, camera in enumerate((camera_0, camera_1)):
        prefix = f"Cam{camera_index}"
        parameters.update(
            {
                f"{prefix}_Fx [pixels]": float(
                    camera.focal_length / camera.pixels_size[0]
                ),
                f"{prefix}_Fy [pixels]": float(
                    camera.focal_length / camera.pixels_size[1]
                ),
                f"{prefix}_Fs [pixels]": 0.0,
                f"{prefix}_Kappa 1": camera.distortion_k1,
                f"{prefix}_Kappa 2": camera.distortion_k2,
                f"{prefix}_Kappa 3": camera.distortion_k3,
                f"{prefix}_P1": camera.distortion_p1,
                f"{prefix}_P2": camera.distortion_p2,
                f"{prefix}_Cx [pixels]": float(camera.c0),
                f"{prefix}_Cy [pixels]": float(camera.c1),
            }
        )

    parameters.update(
        {
            "Tx [mm]": float(extrinsics.translation_cam1_in_cam0[0]),
            "Ty [mm]": float(extrinsics.translation_cam1_in_cam0[1]),
            "Tz [mm]": float(extrinsics.translation_cam1_in_cam0[2]),
            "Theta [deg]": float(rotation[0]),
            "Phi [deg]": float(rotation[1]),
            "Psi [deg]": float(rotation[2]),
        }
    )
    return parameters


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
    "StereoAngles",
    "StereoCameras",
    "StereoExtrinsics",
    "average_subpixel_image",
    "cam_frame_mesh",
    "cam_frame_points",
    "cam_frame_scene",
    "cam_look_at",
    "cam_pos_fill_frame",
    "cam_project_points",
    "crop_image_rectangle",
    "pixel_grid_leng",
    "pixel_vec_leng",
    "subpixel_grid_leng",
    "subpixel_vec_leng",
    "stereo_build_faceon",
    "stereo_build_from_calibration",
    "stereo_build_symmetric",
    "stereo_calc_angles",
    "stereo_calc_baseline",
    "stereo_calc_extrinsics",
    "stereo_calc_stand_off",
    "stereo_save_calibration_matchid",
    "stereo_save_calibration_yaml",
]
