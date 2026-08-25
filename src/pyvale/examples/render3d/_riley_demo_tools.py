"""Run Riley's packaged demonstration scenarios through ``pyvale.render``."""

from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

import pyvale.data as dataset
import pyvale.render as render
import riley


@contextmanager
def use_pyvale_render_api() -> Iterator[None]:
    """Route packaged Riley demo raster calls and data through pyvale.

    The demo modules retain their native Riley mesh and shader construction.
    This adapter changes only their data locations and raster entry point.
    """
    native_raster = riley.raster
    data_paths = {
        "speckle_texture_path": dataset.riley_speckle_texture_path,
        "cal_target_texture_path": dataset.riley_cal_target_texture_path,
        "sphere200_case_path": dataset.riley_sphere200_case_path,
        "platehole_csv_case_path": dataset.riley_platehole_csv_case_path,
        "platehole_exodus_path": dataset.riley_platehole_exodus_path,
        "stereocal_case_path": dataset.riley_stereocal_case_path,
        "rabbit_case_path": dataset.riley_rabbit_case_path,
    }
    original_paths = {
        name: getattr(riley.data, name)
        for name in data_paths
    }

    def raster_through_pyvale(
        meshes: object,
        cameras: object,
        config: riley.RasterConfig,
        out_dir: str | None = None,
    ) -> object:
        mesh_list = (meshes,) if isinstance(meshes, riley.Mesh) else tuple(meshes)
        native_cameras = (
            (cameras,) if isinstance(cameras, riley.Camera) else tuple(cameras)
        )
        camera_list = tuple(_camera_from_riley(camera) for camera in native_cameras)
        riley.raster = native_raster
        try:
            result = render.Riley(config, _path_or_none(out_dir)).render(
                render.Scene3D(mesh_list, camera_list),
            )
        finally:
            riley.raster = raster_through_pyvale

        if result.images is None:
            return None
        return result.images.transpose(0, 1, 4, 2, 3)

    for name, path in data_paths.items():
        setattr(riley.data, name, path)
    riley.raster = raster_through_pyvale
    try:
        yield
    finally:
        riley.raster = native_raster
        for name, path in original_paths.items():
            setattr(riley.data, name, path)


def run_demo(demo_main: object) -> None:
    """Run one packaged Riley demo through the public Pyvale render API."""
    with use_pyvale_render_api():
        demo_main()


def _path_or_none(path: str | None) -> Path | None:
    """Convert Riley's optional output path to the public renderer type."""
    return None if path is None else Path(path)


def _camera_from_riley(camera: riley.Camera) -> render.Camera:
    """Convert a packaged Riley camera to its public Pyvale counterpart."""
    return render.Camera(
        pixels_num=np.asarray(camera.pixels_num),
        pixels_size=np.asarray(camera.pixels_size),
        pos_world=np.asarray(camera.pos_world),
        rot_world=Rotation.from_euler("xyz", camera.rot_world),
        roi_cent_world=np.asarray(camera.roi_cent_world),
        focal_length=camera.focal_length,
        sub_sample=camera.sub_sample,
        distortion_model=camera.distortion_model,
        distortion_k1=camera.distortion_k1,
        distortion_k2=camera.distortion_k2,
        distortion_k3=camera.distortion_k3,
        distortion_k4=camera.distortion_k4,
        distortion_k5=camera.distortion_k5,
        distortion_k6=camera.distortion_k6,
        distortion_p1=camera.distortion_p1,
        distortion_p2=camera.distortion_p2,
        psf_type=camera.psf_type,
        psf_sigma_x=camera.psf_sigma_x,
        psf_sigma_y=camera.psf_sigma_y,
        psf_theta=camera.psf_theta,
        psf_support_rad=camera.psf_support_rad,
    )
