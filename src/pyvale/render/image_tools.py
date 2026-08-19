# ============================================================================
# pyvale: the python validation engine
# License: MIT
# Copyright (C) 2026 Sceptical Rabbit (Lloyd Fletcher)
# ============================================================================
"""Generic image helpers used by planar renderers."""

from enum import Enum
from pathlib import Path

import numpy as np
from PIL import Image


class EImageType(Enum):
    """Supported image file extensions."""

    TIFF = ".tiff"
    BMP = ".bmp"


class ImageTools:
    """Image geometry and output helpers."""

    @staticmethod
    def edge_function(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> np.ndarray:
        return ((c[0] - a[0]) * (b[1] - a[1]) -
                (c[1] - a[1]) * (b[0] - a[0]))

    @staticmethod
    def elem_bound_box_low(coord_min: np.ndarray) -> np.ndarray:
        return np.maximum(np.floor(coord_min).astype(np.int32), 0)

    @staticmethod
    def elem_bound_box_high(coord_max: np.ndarray, image_px: int) -> np.ndarray:
        return np.minimum(np.ceil(coord_max).astype(np.int32), image_px)

    @staticmethod
    def get_num_str(im_num: int, width: int) -> str:
        return str(im_num).zfill(width)

    @staticmethod
    def save_image(path: Path, image: np.ndarray, image_type: EImageType,
                   bits: int) -> None:
        dtype = np.uint8 if bits <= 8 else np.uint16
        Image.fromarray(image[::-1, :].astype(dtype)).save(path.with_suffix(image_type.value))


__all__ = ["EImageType", "ImageTools"]
