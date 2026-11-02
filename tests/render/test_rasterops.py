# ==============================================================================
# pyvale: the python validation engine
# License: MIT
# Copyright (C) 2026 Sceptical Rabbit (Lloyd Fletcher)
# ==============================================================================
"""Analytic tests for low-level raster operations."""

import numpy as np
import pytest

from pyvale.render.rasterops import (
    calculate_edge_function,
    calculate_elem_bound_box_high,
    calculate_elem_bound_box_low,
    format_image_number,
)


def test_edge_function_signs_and_zero() -> None:
    """Verify positive, negative, and collinear points for 2D edge function."""
    point_a = np.array([0.0, 0.0])
    point_b = np.array([2.0, 0.0])

    # Counter-clockwise / left side
    point_left = np.array([1.0, 1.0])
    val_left = calculate_edge_function(point_a, point_b, point_left)
    assert val_left < 0.0

    # Clockwise / right side
    point_right = np.array([1.0, -1.0])
    val_right = calculate_edge_function(point_a, point_b, point_right)
    assert val_right > 0.0

    # Collinear / exactly on the edge
    point_on = np.array([1.0, 0.0])
    val_on = calculate_edge_function(point_a, point_b, point_on)
    assert np.isclose(val_on, 0.0)


def test_elem_bound_box_low() -> None:
    """Lower pixel bound floors coordinates and clips negative values to 0."""
    coords = np.array([[-1.5, 0.0, 2.3, 5.9]])
    low = calculate_elem_bound_box_low(coords)
    np.testing.assert_array_equal(low, np.array([[0, 0, 2, 5]]))


def test_elem_bound_box_high() -> None:
    """Upper pixel bound ceils coordinates and clips to image dimension."""
    coords = np.array([[1.2, 3.0, 10.5, 12.0]])
    high = calculate_elem_bound_box_high(coords, image_px=10)
    np.testing.assert_array_equal(high, np.array([[2, 3, 10, 10]]))


def test_format_image_number() -> None:
    """Verify zero-padding format."""
    assert format_image_number(0, width=4) == "0000"
    assert format_image_number(42, width=4) == "0042"
    assert format_image_number(1234, width=4) == "1234"
