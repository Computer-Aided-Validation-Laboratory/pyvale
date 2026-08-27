"""Generic image helpers used by planar renderers."""

from enum import Enum
from pathlib import Path

import numpy as np
from PIL import Image


class EImageType(Enum):
    """Image file formats supported by :func:`save_image`."""

    TIFF = ".tiff"
    BMP = ".bmp"


def calculate_edge_function(
    point_a: np.ndarray,
    point_b: np.ndarray,
    point_c: np.ndarray,
) -> np.ndarray:
    """Evaluate the signed two-dimensional edge function."""
    return (point_c[0] - point_a[0]) * (point_b[1] - point_a[1]) - (
        point_c[1] - point_a[1]
    ) * (point_b[0] - point_a[0])


def calculate_elem_bound_box_low(coord_min: np.ndarray) -> np.ndarray:
    """Calculate non-negative lower pixel bounds for an element."""
    return np.maximum(np.floor(coord_min).astype(np.int32), 0)


def calculate_elem_bound_box_high(
    coord_max: np.ndarray,
    image_px: int,
) -> np.ndarray:
    """Calculate upper pixel bounds for an element."""
    return np.minimum(np.ceil(coord_max).astype(np.int32), image_px)


def format_image_number(image_number: int, width: int) -> str:
    """Format an image index with leading zeroes."""
    return str(image_number).zfill(width)


def save_image(
    path: Path,
    image: np.ndarray,
    image_type: EImageType,
    bits: int,
) -> None:
    """Save an image in conventional top-to-bottom row orientation."""
    dtype = np.uint8 if bits <= 8 else np.uint16
    image_output = Image.fromarray(image[::-1, :].astype(dtype))
    image_output.save(path.with_suffix(image_type.value))


__all__ = [
    "EImageType",
    "calculate_edge_function",
    "calculate_elem_bound_box_high",
    "calculate_elem_bound_box_low",
    "format_image_number",
    "save_image",
]
