"""Generation and caching of the Riley images used by DIC examples."""

import re
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np
import riley
from scipy.spatial.transform import Rotation

import pyvale.data as dataset
import pyvale.dataio as io

from .camera import Camera
from .mesh import Mesh3D
from .meshtools import mesh_center, meshes3d_from_simdata
from .riley import Riley, to_riley_camera
from .scene import Scene3D
from .uvtools import UVPlane, uv_project_planar_pixels


@dataclass(frozen=True, slots=True)
class DICExampleImages:
    """Paths to example DIC images."""

    cam0_reference: Path
    cam1_reference: Path
    cam0_deformed: Path
    cam1_deformed: Path
    calibration: Path


def _simulation_path(case: str) -> Path:
    if case == "platehole":
        return dataset.dic_platehole_exodus_path()
    displacement = int(case.removeprefix("rigid_").removesuffix("px"))
    return dataset.dic_rigid_exodus_path(displacement)


def _load_mesh(exodus_path: Path) -> tuple[Mesh3D, int]:
    exodus = riley.load_exodus(
        exodus_path,
        connect_keys=("connect1",),
        nodal_keys="all",
    )
    block = exodus.elem_blocks["connect1"]
    disp_x = np.asarray(exodus.nodal_vars["disp_x"], dtype=np.float64)
    disp_y = np.asarray(exodus.nodal_vars["disp_y"], dtype=np.float64)
    disp_z = np.asarray(
        exodus.nodal_vars.get("disp_z", np.zeros_like(disp_x)),
        dtype=np.float64,
    )
    simulation = io.SimData(
        coords=exodus.coords,
        connect={"connect1": block.connect},
        node_vars={"disp_x": disp_x, "disp_y": disp_y, "disp_z": disp_z},
    )
    mesh = meshes3d_from_simdata(
        simulation,
        {
            "connect1": riley.ConnectConvention(
                block.elem_type,
                riley.EConnectAxis.ROW,
                1,
                riley.ENodeOrder.EXODUS,
            )
        },
        displacement_keys=("disp_x", "disp_y", "disp_z"),
    )["connect1"]
    return mesh, mesh.displacements.shape[0]


def _paths(case: str, output_dir: Path) -> DICExampleImages:
    prefix = "platehole" if case == "platehole" else "rigid"
    return DICExampleImages(
        cam0_reference=output_dir / f"{prefix}_cam0_frame00.tiff",
        cam1_reference=output_dir / f"{prefix}_cam1_frame00.tiff",
        cam0_deformed=output_dir / f"{prefix}_cam0_frame*.tiff",
        cam1_deformed=output_dir / f"{prefix}_cam1_frame*.tiff",
        calibration=output_dir / "stereo_data_opencv.csv",
    )


def _complete(
    case: str,
    paths: DICExampleImages,
    output_dir: Path,
    frame_labels: tuple[int, ...],
) -> bool:
    """Return whether the requested frames are already present.

    A rigid-image directory contains frames from two simulations, so counting
    every matching frame is not sufficient to determine whether one of the
    simulations is cached.
    """
    prefix = "platehole" if case == "platehole" else "rigid"
    expected_cam0 = tuple(
        output_dir / f"{prefix}_cam0_frame{label:02d}.tiff" for label in frame_labels
    )
    expected_cam1 = tuple(
        output_dir / f"{prefix}_cam1_frame{label:02d}.tiff" for label in frame_labels
    )
    return (
        paths.cam0_reference.is_file()
        and paths.cam1_reference.is_file()
        and paths.calibration.is_file()
        and all(path.is_file() for path in expected_cam0)
        and all(path.is_file() for path in expected_cam1)
    )


def _rename_images(case: str, output_dir: Path, frame_labels: tuple[int, ...]) -> None:
    prefix = "platehole" if case == "platehole" else "rigid"
    for camera in (0, 1):
        sources = sorted(
            output_dir.glob(f"cam{camera}_frame*_field0.tiff"),
            key=lambda path: int(
                re.search(r"frame(\d+)_field0\.tiff$", path.name).group(1)
            ),
        )
        for source, label in zip(sources, frame_labels, strict=True):
            target = output_dir / f"{prefix}_cam{camera}_frame{label:02d}.tiff"
            source.replace(target)


def _render_case(
    case: str,
    output_dir: Path,
    *,
    frame_labels: tuple[int, ...] | None = None,
    frame_indices: tuple[int, ...] | None = None,
) -> DICExampleImages:

    mesh, num_frames = _load_mesh(_simulation_path(case))
    if frame_indices is not None:
        mesh.displacements = mesh.displacements[list(frame_indices)]
        num_frames = len(frame_indices)
    if frame_labels is None:
        frame_labels = tuple(range(num_frames))
    paths = _paths(case, output_dir)
    if _complete(case, paths, output_dir, frame_labels):
        return paths

    texture = riley.load_texture_mono_u8(dataset.dic_pattern_5mpx_path())
    pixels_num = (1040, 1540)
    pixels_size = (3.45e-6, 3.45e-6)
    focal_length = 50.0e-3
    leng_per_px = 0.1e-3
    px_per_spec = 5.0
    texture_shape = (texture.shape[1], texture.shape[2])

    # Match the benchmark scale: 5 texture pixels per speckle and
    # 0.1 mm per output pixel. The resulting specimen is inset by 20 px.
    scale = px_per_spec / (px_per_spec * leng_per_px)
    x_min, y_min = np.min(mesh.coords[:, :2], axis=0)
    x_max, y_max = np.max(mesh.coords[:, :2], axis=0)
    px_w = (x_max - x_min) * scale
    px_h = (y_max - y_min) * scale
    px_bbox = (
        0.5 * (texture_shape[1] - px_w),
        0.5 * (texture_shape[0] - px_h),
        0.5 * (texture_shape[1] + px_w),
        0.5 * (texture_shape[0] + px_h),
    )
    mesh.shader = riley.TextureShader(
        uvs=uv_project_planar_pixels(
            mesh.coords,
            texture_shape,
            px_bbox,
            plane=UVPlane(
                normal=np.array((0.0, 0.0, -1.0)),
                origin=np.zeros(3),
            ),
        ),
        texture=texture,
    )

    roi_centre = np.asarray(mesh_center(mesh))
    z_dist = focal_length / (pixels_size[0] / leng_per_px)
    stereo_angle = np.deg2rad(20.0)
    rotations = [(0.0, 0.0, 0.0), (0.0, stereo_angle, 0.0)]
    positions = [
        (roi_centre[0], roi_centre[1], roi_centre[2] + z_dist),
        (
            roi_centre[0] + z_dist * np.sin(stereo_angle),
            roi_centre[1],
            roi_centre[2] + z_dist * np.cos(stereo_angle),
        ),
    ]
    cameras: list[Camera] = []
    for rotation, position in zip(rotations, positions, strict=True):
        cameras.append(
            Camera(
                pixels_num=pixels_num,
                pixels_size=pixels_size,
                pos_world=position,
                rot_world=Rotation.from_euler("xyz", rotation),
                roi_cent_world=roi_centre,
                focal_length=focal_length,
                subsample=4,
            )
        )

    config = riley.create_raster_config(
        num_frames=num_frames, total_threads=8, save_strategy=riley.SaveStrategy.disk
    )
    config.background_value = 128.0
    config.tile_size_max = 128
    config.save_scaling = riley.ScaleStrategy.none
    config.save_format = riley.ImageFormat.tiff

    output_dir.mkdir(parents=True, exist_ok=True)
    Riley(config, output_dir).render(Scene3D(meshes=[mesh], cameras=cameras))
    _rename_images(case, output_dir, frame_labels)
    riley.save_stereo_pair(
        str(output_dir),
        "stereo_data_opencv.csv",
        replace(to_riley_camera(cameras[0]), coord_sys=riley.CameraCoordSys.opencv),
        replace(to_riley_camera(cameras[1]), coord_sys=riley.CameraCoordSys.opencv),
    )
    return paths


def create_example_images_platehole() -> DICExampleImages:
    """Create and cache Riley images for the plate-hole DIC examples.

    Returns
    -------
    DICExampleImages
        Paths to the generated reference, deformed, and calibration files.
    """
    output_dir = Path.cwd() / "pyvale-output" / "dic_images" / "platehole"
    return _render_case("platehole", output_dir)


def create_example_images_rigid() -> DICExampleImages:
    """Create and cache rigid images in one folder.

    Frames 00-10 are the 1px case; frames 25 and 50 are frames 1 and 2
    from the 50px simulation.

    Returns
    -------
    DICExampleImages
        Paths to the generated reference, deformed, and calibration files.
    """
    output_dir = Path.cwd() / "pyvale-output" / "dic_images" / "rigid"
    one_px = _render_case("rigid_1px", output_dir, frame_labels=tuple(range(11)))
    _render_case(
        "rigid_50px",
        output_dir,
        frame_labels=(25, 50),
        frame_indices=(1, 2),
    )

    return DICExampleImages(
        cam0_reference=one_px.cam0_reference,
        cam1_reference=one_px.cam1_reference,
        cam0_deformed=one_px.cam0_deformed,
        cam1_deformed=one_px.cam1_deformed,
        calibration=one_px.calibration,
    )
