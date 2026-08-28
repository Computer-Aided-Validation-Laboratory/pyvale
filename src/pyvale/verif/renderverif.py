#===============================================================================
# pyvale: the python validation engine
# License: MIT
# Copyright (C) 2026 Sceptical Rabbit (Lloyd Fletcher)
#===============================================================================

"""
DEVELOPER VERIFICATION MODULE
--------------------------------------------------------------------------------
This module contains developer utility functions used for verification testing
of the render toolbox in pyvale.

Specifically, this module contains hard-coded verification scenes, packaged-
case loaders shared between the gold generation scripts and the render tests,
and image-regression assertions.
"""

from pathlib import Path
import re

import numpy as np
from scipy.spatial.transform import Rotation

import pyvale.data as dataset
import pyvale.dataio as io
import pyvale.mooseherder as mooseherder
import pyvale.render as render
import pyvale.sensorsim as sensorsim
import riley


def assert_render_allclose(
    actual: np.ndarray,
    reference: np.ndarray,
    case_ident: str,
    *,
    rtol: float = 1.0e-9,
    atol: float = 1.0e-9,
) -> None:
    """Assert image equality and save useful diagnostics when it fails.

    Parameters
    ----------
    actual, reference : numpy.ndarray
        Rendered and trusted reference image arrays with matching shapes.
    case_ident : str
        Stable case identifier used for the failure output directory.
    rtol, atol : float, optional
        Relative and absolute tolerances passed to :func:`numpy.allclose`.

    Raises
    ------
    AssertionError
        If arrays differ. Raw NumPy diagnostics are saved to
        ``render-fails/<case_ident>`` before the error is raised. Scaled TIFF
        images are also saved when Pillow is installed.
    """
    if np.allclose(actual, reference, rtol=rtol, atol=atol):
        return

    directory = Path("render-fails") / _safe_case_ident(case_ident)
    directory.mkdir(parents=True, exist_ok=True)

    difference = np.asarray(actual, dtype=np.float64) - np.asarray(
        reference, dtype=np.float64,
    )
    np.save(directory / "render.npy", actual)
    np.save(directory / "reference.npy", reference)
    np.save(directory / "difference.npy", difference)

    if _pil_image() is not None:
        actual_image, reference_image, difference_image = _select_images(
            actual, reference, difference,
        )
        lower = min(np.nanmin(actual_image), np.nanmin(reference_image))
        upper = max(np.nanmax(actual_image), np.nanmax(reference_image))
        _save_tiff(directory / "render.tiff", actual_image, lower, upper)
        _save_tiff(directory / "reference.tiff", reference_image, lower, upper)
        _save_tiff(
            directory / "difference.tiff", difference_image,
            float(np.nanmin(difference_image)), float(np.nanmax(difference_image)),
        )

    maximum = float(np.nanmax(np.abs(difference)))
    raise AssertionError(
        f"Render mismatch for {case_ident}; maximum absolute difference "
        f"is {maximum:.6e}. Diagnostics: {directory}",
    )


def _pil_image() -> object | None:
    """Return the Pillow Image class or None when Pillow is not installed."""
    try:
        from PIL import Image
    except ImportError:
        return None
    return Image


def _safe_case_ident(case_ident: str) -> str:
    """Return a portable directory name from a descriptive case identifier."""
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", case_ident).strip("_")


def _select_images(
    actual: np.ndarray,
    reference: np.ndarray,
    difference: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Select the two-dimensional plane containing the largest difference."""
    if difference.ndim == 2:
        return actual, reference, difference
    if difference.ndim < 2:
        return actual.reshape(1, -1), reference.reshape(1, -1), difference.reshape(1, -1)

    image_shape = difference.shape[-2:]
    difference_planes = difference.reshape((-1, *image_shape))
    plane_index = int(np.argmax(np.max(np.abs(difference_planes), axis=(1, 2))))
    return (
        actual.reshape((-1, *image_shape))[plane_index],
        reference.reshape((-1, *image_shape))[plane_index],
        difference_planes[plane_index],
    )


def _save_tiff(path: Path, image: np.ndarray, lower: float, upper: float) -> None:
    """Save one array as a full-range unsigned 8-bit diagnostic TIFF."""
    image_class = _pil_image()
    if image_class is None:
        return
    if not np.isfinite(lower) or not np.isfinite(upper) or upper <= lower:
        scaled = np.zeros(image.shape, dtype=np.uint8)
    else:
        scaled = np.clip((image - lower) / (upper - lower), 0.0, 1.0)
        scaled = np.rint(255.0 * scaled).astype(np.uint8)
    image_class.fromarray(scaled).save(path)


def render_triangle(output_dir: Path) -> np.ndarray:
    """Render the deterministic common-API Blender triangle scene."""
    mesh = render.Mesh3D(
        render.EElementType.TRI3,
        np.array(((-1.0, -1.0, 0.0), (1.0, -1.0, 0.0),
                  (0.0, 1.0, 0.0))),
        np.array(((0, 1, 2),)), object(),
    )
    camera = render.Camera(
        pixels_num=np.array((32, 32)),
        pixels_size=np.array((0.02, 0.02)),
        pos_world=np.array((0.0, 0.0, 2.0)),
        rot_world=Rotation.identity(),
        roi_cent_world=np.zeros(3),
        focal_length=1.0,
    )
    result = render.Blender(render.BlenderConfig(output_dir, samples=1)).render(
        render.Scene3D([mesh], [camera]),
    )
    assert result.images is not None
    return result.images


def riley_memory_config() -> riley.RasterConfig:
    """Return a single-frame Riley raster configuration kept in memory."""
    return riley.create_raster_config(
        1, save_strategy=riley.SaveStrategy.memory,
    )


def riley_rabbit_scene() -> render.Scene3D:
    """Build the committed multi-mesh rabbit regression scene.

    Returns
    -------
    render.Scene3D
        The packaged TRI3 rabbit meshes viewed by a camera filled from
        their combined extent.
    """
    meshes = dataset.riley_rabbit_meshes()
    coords = np.concatenate([mesh.coords for mesh in meshes])
    pixels_num = np.array((320, 160))
    pixels_size = np.array((5.3e-6, 5.3e-6))
    focal_length = 50.0e-3
    rotation = Rotation.identity()
    position = riley.pos_fill_frame_from_rot(
        coords, tuple(pixels_num), tuple(pixels_size), focal_length,
        tuple(rotation.as_euler("xyz")), 1.1,
    )
    camera = render.Camera(
        pixels_num=pixels_num,
        pixels_size=pixels_size,
        pos_world=np.asarray(position),
        rot_world=rotation,
        roi_cent_world=np.mean(coords, axis=0),
        focal_length=focal_length,
    )
    return render.Scene3D(meshes, [camera])


def scaled_mechanical_2d() -> io.SimData:
    """Load the mechanical 2D case scaled to millimetres for Blender scenes."""
    sim_data = mooseherder.ExodusLoader(
        dataset.mechanical_2d_path(),
    ).load_all_sim_data()
    sensorsim.scale_length_units(1000.0, sim_data, ("disp_x", "disp_y"))
    return sim_data


__all__ = [
    "assert_render_allclose",
    "render_triangle",
    "riley_memory_config",
    "riley_rabbit_scene",
    "scaled_mechanical_2d",
]
