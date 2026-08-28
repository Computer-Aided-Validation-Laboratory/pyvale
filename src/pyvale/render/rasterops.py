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


__all__ = [
    "calculate_edge_function",
    "calculate_elem_bound_box_high",
    "calculate_elem_bound_box_low",
    "format_image_number",
]
