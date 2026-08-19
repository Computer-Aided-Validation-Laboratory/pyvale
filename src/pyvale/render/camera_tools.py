# ============================================================================
# pyvale: the python validation engine
# License: MIT
# Copyright (C) 2026 Sceptical Rabbit (Lloyd Fletcher)
# ============================================================================
"""Generic orthographic camera-grid operations for image warping."""

import numpy as np
from scipy.signal import convolve2d


class CameraTools:
    """Small camera-grid helpers shared by planar renderers."""

    @staticmethod
    def pixel_vec_leng(field_of_view: np.ndarray,
                       leng_per_px: float) -> tuple[np.ndarray, np.ndarray]:
        return (
            np.arange(leng_per_px / 2.0, field_of_view[0], leng_per_px),
            np.arange(leng_per_px / 2.0, field_of_view[1], leng_per_px),
        )

    @staticmethod
    def pixel_grid_leng(field_of_view: np.ndarray,
                        leng_per_px: float) -> tuple[np.ndarray, np.ndarray]:
        return np.meshgrid(*CameraTools.pixel_vec_leng(field_of_view, leng_per_px))

    @staticmethod
    def subpixel_vec_leng(field_of_view: np.ndarray,
                          leng_per_px: float,
                          subsample: int) -> tuple[np.ndarray, np.ndarray]:
        spacing = leng_per_px / subsample
        return (
            np.arange(spacing / 2.0, field_of_view[0], spacing),
            np.arange(spacing / 2.0, field_of_view[1], spacing),
        )

    @staticmethod
    def subpixel_grid_leng(field_of_view: np.ndarray,
                           leng_per_px: float,
                           subsample: int) -> tuple[np.ndarray, np.ndarray]:
        return np.meshgrid(
            *CameraTools.subpixel_vec_leng(field_of_view, leng_per_px, subsample),
        )

    @staticmethod
    def crop_image_rectangle(image: np.ndarray,
                             pixels_count: np.ndarray) -> np.ndarray:
        """Crop an image to its camera extent from the upper-left corner."""
        return image[:pixels_count[1], :pixels_count[0]]

    @staticmethod
    def average_subpixel_image(image: np.ndarray, subsample: int) -> np.ndarray:
        """Average an image's square subpixel blocks."""
        if subsample <= 1:
            return image
        kernel = np.ones((subsample, subsample)) / (subsample ** 2)
        convolved = convolve2d(image, kernel, mode="same")
        start = round(subsample / 2.0) - 1
        return convolved[start::subsample, start::subsample]


__all__ = ["CameraTools"]
