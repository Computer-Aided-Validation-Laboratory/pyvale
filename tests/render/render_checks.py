"""Shared image-regression assertions for render tests."""

from pathlib import Path
import re

import numpy as np
from PIL import Image


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
        If arrays differ. Raw NumPy and scaled TIFF diagnostics are saved to
        ``render-fails/<case_ident>`` before the error is raised.
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
    if not np.isfinite(lower) or not np.isfinite(upper) or upper <= lower:
        scaled = np.zeros(image.shape, dtype=np.uint8)
    else:
        scaled = np.clip((image - lower) / (upper - lower), 0.0, 1.0)
        scaled = np.rint(255.0 * scaled).astype(np.uint8)
    Image.fromarray(scaled).save(path)


__all__ = ["assert_render_allclose"]
