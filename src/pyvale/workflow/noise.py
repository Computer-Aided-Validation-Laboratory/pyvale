"""Reusable image noise utilities for workflow functions."""

import numpy as np


def add_grey_level_noise(
    image: np.ndarray,
    rng: np.random.Generator,
    standard_deviation_fraction: float = 0.005,
) -> np.ndarray:
    """Add independent Gaussian noise, clip, and quantise to input dtype."""
    if standard_deviation_fraction < 0.0:
        raise ValueError("standard_deviation_fraction must be non-negative.")
    if np.issubdtype(image.dtype, np.integer):
        maximum = float(np.iinfo(image.dtype).max)
    else:
        maximum = 1.0
    noisy = image.astype(np.float64) + rng.normal(
        0.0,
        maximum * standard_deviation_fraction,
        image.shape,
    )
    noisy = np.clip(noisy, 0.0, maximum)
    if np.issubdtype(image.dtype, np.integer):
        noisy = np.rint(noisy)
    return noisy.astype(image.dtype)
