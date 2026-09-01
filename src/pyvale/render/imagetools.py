# ==============================================================================
# pyvale: the python validation engine
# License: MIT
# Copyright (C) 2026 Sceptical Rabbit (Lloyd Fletcher)
# ==============================================================================
"""User facing image and texture loading, saving, and preparation helpers."""

from enum import Enum
from pathlib import Path

import numpy as np
from PIL import Image


class EImageType(Enum):
    """Image file formats supported by :func:`image_save`."""

    TIFF = ".tiff"
    BMP = ".bmp"
    PNG = ".png"


def image_load(path: Path | str) -> np.ndarray:
    """Load an image from disk as a numpy array.

    Parameters
    ----------
    path : pathlib.Path or str
        Filesystem path to the image.

    Returns
    -------
    np.ndarray
        Image array with shape ``(height, width)`` for greyscale or
        ``(height, width, num_channels)`` for colour images.
    """
    img_path = Path(path)
    if not img_path.is_file():
        raise FileNotFoundError(f"Image file does not exist: {img_path}")

    with Image.open(img_path) as pil_img:
        return np.asarray(pil_img)


def image_save(
    path: Path | str,
    image: np.ndarray,
    image_type: EImageType = EImageType.TIFF,
    bits: int = 8,
) -> None:
    """Save an image in conventional top to bottom row orientation.

    Parameters
    ----------
    path : pathlib.Path or str
        Destination filesystem path.
    image : np.ndarray
        Image array with shape ``(height, width)`` or
        ``(height, width, num_channels)``.
    image_type : EImageType, optional
        Format extension (default is ``EImageType.TIFF``).
    bits : int, optional
        Bit depth (8 or 16). Defaults to 8.
    """
    out_path = Path(path)
    if out_path.suffix != image_type.value:
        out_path = out_path.with_suffix(image_type.value)

    dtype = np.uint8 if bits <= 8 else np.uint16
    img_arr = np.asarray(image, dtype=dtype)

    out_img = Image.fromarray(img_arr[::-1, :])
    out_img.save(out_path)


def image_crop(
    image: np.ndarray,
    x: int,
    y: int,
    width: int,
    height: int,
) -> np.ndarray:
    """Crop a rectangular region from an image (top left origin).

    Parameters
    ----------
    image : np.ndarray
        Input image array with shape ``(height, width)`` or
        ``(height, width, num_channels)``.
    x, y : int
        Top left pixel coordinates of the crop rectangle.
    width, height : int
        Width and height of the crop region in pixels.

    Returns
    -------
    np.ndarray
        Cropped subarray with shape ``(height, width, ...)`` and matching
        dtype.

    Raises
    ------
    ValueError
        If crop bounds are invalid or exceed image dimensions.
    """
    img_h, img_w = image.shape[:2]

    if x < 0 or y < 0 or width <= 0 or height <= 0:
        raise ValueError(
            f"Invalid crop dimensions (x={x}, y={y}, w={width}, h={height})."
        )
        
    if x + width > img_w or y + height > img_h:
        raise ValueError(
            f"Crop bounds (x={x}+{width}, y={y}+{height}) exceed image shape "
            f"({img_w}x{img_h})."
        )

    return image[y : y + height, x : x + width].copy()


def image_resize(
    image: np.ndarray,
    size: tuple[int, int],
) -> np.ndarray:
    """Resize an image to ``(width, height)`` in pixels.

    Parameters
    ----------
    image : np.ndarray
        Input image array with shape ``(height, width)`` or
        ``(height, width, num_channels)``.
    size : tuple[int, int]
        Target ``(width, height)`` dimensions in pixels.

    Returns
    -------
    np.ndarray
        Resized array with shape ``(size[1], size[0], ...)`` matching input
        dtype.
    """
    orig_dtype = image.dtype
    with Image.fromarray(image) as pil_img:
        resized = pil_img.resize(size, resample=Image.Resampling.BILINEAR)
        return np.asarray(resized, dtype=orig_dtype)


def image_normalise(
    image: np.ndarray,
    lower: float = 0.0,
    upper: float = 1.0,
) -> np.ndarray:
    """Normalise finite pixel intensities onto the range ``[lower, upper]``.

    Parameters
    ----------
    image : np.ndarray
        Input image array with shape ``(height, width)`` or
        ``(height, width, num_channels)``.
    lower : float, optional
        Target lower intensity value. Defaults to 0.0.
    upper : float, optional
        Target upper intensity value. Defaults to 1.0.

    Returns
    -------
    np.ndarray
        Normalised image array with shape identical to input and dtype
        ``float64``.
    """
    arr = np.asarray(image, dtype=np.float64)
    min_val = np.nanmin(arr)
    max_val = np.nanmax(arr)

    if np.isclose(min_val, max_val):
        return np.full_like(arr, lower, dtype=np.float64)

    norm = (arr - min_val) / (max_val - min_val)
    return lower + norm * (upper - lower)


def image_grayscale(image: np.ndarray) -> np.ndarray:
    """Convert an RGB or RGBA image to greyscale using standard luminance.

    Parameters
    ----------
    image : np.ndarray
        Input image array with shape ``(height, width)`` or
        ``(height, width, num_channels)``.

    Returns
    -------
    np.ndarray
        Greyscale image array with shape ``(height, width)`` and dtype
        ``float64``.

    Raises
    ------
    ValueError
        If the input image array shape is unsupported.
    """
    arr = np.asarray(image, dtype=np.float64)
    if arr.ndim == 2:
        return arr.copy()

    if arr.ndim == 3:
        if arr.shape[2] >= 3:
            return (
                0.299 * arr[:, :, 0]
                + 0.587 * arr[:, :, 1]
                + 0.114 * arr[:, :, 2]
            )
            
    raise ValueError(
        f"Unsupported image shape for grayscale conversion: {arr.shape}"
    )


__all__ = [
    "EImageType",
    "image_crop",
    "image_grayscale",
    "image_load",
    "image_normalise",
    "image_resize",
    "image_save",
]
