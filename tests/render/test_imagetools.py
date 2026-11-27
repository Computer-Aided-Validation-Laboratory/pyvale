# ==============================================================================
# pyvale: the python validation engine
# License: MIT
# Copyright (C) 2026 Sceptical Rabbit (Lloyd Fletcher)
# ==============================================================================
"""Analytic tests for image loading, saving, cropping, normalisation."""

from pathlib import Path

import numpy as np
import pytest

import pyvale.render as render


def test_image_crop_analytic() -> None:
    """Cropping a grid subarray extracts exact pixel values."""
    img = np.arange(25, dtype=np.uint8).reshape(5, 5)
    cropped = render.image_crop(img, x=1, y=2, width=3, height=2)

    expected = np.array([
        [11, 12, 13],
        [16, 17, 18],
    ], dtype=np.uint8)
    np.testing.assert_array_equal(cropped, expected)


def test_image_crop_out_of_bounds_raises() -> None:
    """Out-of-bounds crops raise ValueError."""
    img = np.zeros((10, 10), dtype=np.uint8)
    with pytest.raises(ValueError, match="exceed image shape"):
        render.image_crop(img, x=8, y=8, width=5, height=5)


def test_image_normalise() -> None:
    """Min-max intensity scaling maps to specified range."""
    img = np.array([0.0, 5.0, 10.0])
    norm = render.image_normalise(img, lower=0.0, upper=1.0)
    np.testing.assert_allclose(norm, np.array([0.0, 0.5, 1.0]))

    custom_norm = render.image_normalise(img, lower=100.0, upper=200.0)
    np.testing.assert_allclose(custom_norm, np.array([100.0, 150.0, 200.0]))


def test_image_normalise_constant_image() -> None:
    """Constant image normalises to lower bound."""
    img = np.full((4, 4), 42.0)
    norm = render.image_normalise(img, lower=0.0, upper=1.0)
    np.testing.assert_allclose(norm, np.zeros((4, 4)))


def test_image_grayscale() -> None:
    """RGB to grayscale uses standard luminance formula."""
    rgb = np.zeros((1, 1, 3), dtype=np.uint8)
    rgb[0, 0] = [255, 0, 0]  # pure red
    grey = render.image_grayscale(rgb)
    assert np.isclose(grey[0, 0], 0.299 * 255.0)


def test_image_save_and_load_roundtrip(tmp_path: Path) -> None:
    """Image save and load round-trip preserves values."""
    img = np.arange(100, dtype=np.uint8).reshape(10, 10)
    out_file = tmp_path / "test_roundtrip.tiff"

    render.image_save(out_file, img, render.EImageType.TIFF, bits=8)
    loaded = render.image_load(out_file)

    # Note: image_save flips vertically for standard image orientation
    # image_load loads standard orientation
    np.testing.assert_array_equal(loaded, img[::-1, :])
