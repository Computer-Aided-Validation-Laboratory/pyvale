# ==============================================================================
# pyvale: the python validation engine
# License: MIT
# Copyright (C) 2026 Sceptical Rabbit (Lloyd Fletcher)
# ==============================================================================
"""Private rasterisation helpers used by 2D planar deformation renderers."""

import numpy as np


def calculate_edge_function(
    point_a: np.ndarray,
    point_b: np.ndarray,
    point_c: np.ndarray,
) -> np.ndarray:
    """Evaluate the signed two dimensional edge function.

    Parameters
    ----------
    point_a : np.ndarray
        First vertex coordinate array with shape ``(2,)`` and dtype
        ``float64`` representing (X, Y).
    point_b : np.ndarray
        Second vertex coordinate array with shape ``(2,)`` and dtype
        ``float64`` representing (X, Y).
    point_c : np.ndarray
        Evaluation point coordinate array with shape ``(2,)`` or ``(2, N)``
        and dtype ``float64`` representing (X, Y).

    Returns
    -------
    np.ndarray
        Signed edge function value(s) with dtype ``float64``.
    """
    return (point_c[0] - point_a[0]) * (point_b[1] - point_a[1]) - (
        point_c[1] - point_a[1]
    ) * (point_b[0] - point_a[0])


def calculate_elem_bound_box_low(coord_min: np.ndarray) -> np.ndarray:
    """Calculate non negative lower pixel bounds for an element.

    Parameters
    ----------
    coord_min : np.ndarray
        Minimum coordinate array with shape ``(2,)`` and dtype ``float64``
        representing (min_x, min_y).

    Returns
    -------
    np.ndarray
        Lower pixel bounding box array with shape ``(2,)`` and dtype ``int32``
        clamped to a minimum of 0.
    """
    return np.maximum(np.floor(coord_min).astype(np.int32), 0)


def calculate_elem_bound_box_high(
    coord_max: np.ndarray,
    image_px: int,
) -> np.ndarray:
    """Calculate upper pixel bounds for an element.

    Parameters
    ----------
    coord_max : np.ndarray
        Maximum coordinate array with shape ``(2,)`` and dtype ``float64``
        representing (max_x, max_y).
    image_px : int
        Maximum image dimension in pixels.

    Returns
    -------
    np.ndarray
        Upper pixel bounding box array with shape ``(2,)`` and dtype ``int32``
        clamped to ``image_px``.
    """
    return np.minimum(np.ceil(coord_max).astype(np.int32), image_px)


def format_image_number(image_number: int, width: int) -> str:
    """Format an image index with leading zeroes.

    Parameters
    ----------
    image_number : int
        Zero based image index.
    width : int
        Total character width for zero padding.

    Returns
    -------
    str
        Zero padded image number string.
    """
    return str(image_number).zfill(width)


__all__ = [
    "calculate_edge_function",
    "calculate_elem_bound_box_high",
    "calculate_elem_bound_box_low",
    "format_image_number",
]
