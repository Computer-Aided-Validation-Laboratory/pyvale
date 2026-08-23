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
    """Image file formats supported by :meth:`ImageTools.save_image`."""

    TIFF = ".tiff"
    BMP = ".bmp"


class ImageTools:
    """Image geometry and output helpers for planar renderers."""

    @staticmethod
    def edge_function(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> np.ndarray:
        """Evaluate the signed two-dimensional edge function.

        Parameters
        ----------
        a, b, c : numpy.ndarray
            Two-dimensional points or compatible arrays of points.

        Returns
        -------
        numpy.ndarray
            Signed cross-product value for the edge from ``a`` to ``b``.
        """
        return ((c[0] - a[0]) * (b[1] - a[1]) -
                (c[1] - a[1]) * (b[0] - a[0]))

    @staticmethod
    def elem_bound_box_low(coord_min: np.ndarray) -> np.ndarray:
        """Calculate clipped lower pixel bounds for an element.

        Parameters
        ----------
        coord_min : numpy.ndarray
            Minimum element coordinates in image-pixel space.

        Returns
        -------
        numpy.ndarray
            Non-negative integer lower bounds.
        """
        return np.maximum(np.floor(coord_min).astype(np.int32), 0)

    @staticmethod
    def elem_bound_box_high(coord_max: np.ndarray, image_px: int) -> np.ndarray:
        """Calculate clipped upper pixel bounds for an element.

        Parameters
        ----------
        coord_max : numpy.ndarray
            Maximum element coordinates in image-pixel space.
        image_px : int
            Exclusive image bound along the relevant dimension.

        Returns
        -------
        numpy.ndarray
            Integer upper bounds clipped to ``image_px``.
        """
        return np.minimum(np.ceil(coord_max).astype(np.int32), image_px)

    @staticmethod
    def get_num_str(im_num: int, width: int) -> str:
        """Format an image index with leading zeroes.

        Parameters
        ----------
        im_num : int
            Image index to format.
        width : int
            Minimum number of decimal digits.

        Returns
        -------
        str
            Zero-padded decimal representation of ``im_num``.
        """
        return str(im_num).zfill(width)

    @staticmethod
    def save_image(path: Path, image: np.ndarray, image_type: EImageType,
                   bits: int) -> None:
        """Save an image after changing to conventional image row orientation.

        Parameters
        ----------
        path : pathlib.Path
            Output path, whose suffix is replaced by ``image_type``.
        image : numpy.ndarray
            Image array in the render coordinate convention.
        image_type : EImageType
            File type used for output.
        bits : int
            Output bit depth. Values up to eight use unsigned 8-bit storage;
            larger values use unsigned 16-bit storage.
        """
        dtype = np.uint8 if bits <= 8 else np.uint16
        Image.fromarray(image[::-1, :].astype(dtype)).save(path.with_suffix(image_type.value))


__all__ = ["EImageType", "ImageTools"]
