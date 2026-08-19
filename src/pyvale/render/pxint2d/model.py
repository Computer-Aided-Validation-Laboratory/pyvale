# ============================================================================
# pyvale: the python validation engine
# License: MIT
# Copyright (C) 2026 Sceptical Rabbit (Lloyd Fletcher)
# ============================================================================
"""Configuration, quadrature, and texture data for PixInt2D."""

from dataclasses import dataclass, field
from enum import StrEnum

import numpy as np


class EPxIntMapping(StrEnum):
    """Inverse finite-element maps supported by PixInt2D."""

    AFFINE = "affine"
    VTK = "vtk"
    NEWTON_ONE_ELEM = "newton_one_elem"
    NEWTON_MESH_STRUCT = "newton_mesh_struct"
    NEWTON_MESH_UNSTRUCT = "newton_mesh_unstruct"
    STRUCTURED_QUAD9 = "structured_quad9"


@dataclass(frozen=True, slots=True)
class RectRule:
    """Uniform rectangular pixel quadrature.

    Parameters
    ----------
    samples_per_axis : int
        Number of point samples along each pixel axis.
    """

    samples_per_axis: int = 1

    def __post_init__(self) -> None:
        """Validate the square sampling count."""
        if self.samples_per_axis < 1:
            raise ValueError("samples_per_axis must be positive.")


@dataclass(frozen=True, slots=True)
class GaussRule:
    """Tensor-product Gauss-Legendre pixel quadrature.

    Parameters
    ----------
    points_per_axis : int
        Number of Gauss points along each pixel axis.
    """

    points_per_axis: int = 2

    def __post_init__(self) -> None:
        """Validate the square quadrature order."""
        if self.points_per_axis < 1:
            raise ValueError("points_per_axis must be positive.")


@dataclass(frozen=True, slots=True)
class AnalyticRule:
    """Request an exact analytic integral where a renderer supports it."""


@dataclass(frozen=True, slots=True)
class GaussianPSF:
    """Gaussian point-spread function in pixel coordinates.

    Parameters
    ----------
    sigma_pixels : float
        Gaussian standard deviation in pixels.
    support_sigmas : float
        Finite support radius expressed as standard deviations.
    """

    sigma_pixels: float = 1.0
    support_sigmas: float = 4.0

    def __post_init__(self) -> None:
        """Validate the physical PSF parameters."""
        if self.sigma_pixels <= 0.0 or self.support_sigmas <= 0.0:
            raise ValueError("PSF parameters must be positive.")


@dataclass(frozen=True, slots=True)
class PxInt2DOpts:
    """Operational options shared by PixInt2D renderers.

    Parameters
    ----------
    mapping : EPxIntMapping
        Inverse-map implementation used for every quadrature point.
    integration : RectRule, GaussRule, or AnalyticRule
        Pixel integration rule.
    workers : int
        Reserved worker count for future parallel execution.
    max_points_per_chunk : int
        Upper bound on quadrature points processed as one chunk.
    psf : GaussianPSF or None
        Optional output-image PSF.
    """

    mapping: EPxIntMapping = EPxIntMapping.NEWTON_MESH_UNSTRUCT
    integration: RectRule | GaussRule | AnalyticRule = field(
        default_factory=RectRule,
    )
    workers: int = 1
    max_points_per_chunk: int = 500_000
    psf: GaussianPSF | None = None

    def __post_init__(self) -> None:
        """Validate non-physical execution controls."""
        if self.workers < 1 or self.max_points_per_chunk < 1:
            raise ValueError("workers and max_points_per_chunk must be positive.")


@dataclass(frozen=True, slots=True)
class Eggbox:
    """Analytic periodic grid texture in reference-world coordinates.

    Parameters
    ----------
    mean : float
        Mean normalised intensity.
    contrast : float
        Peak-to-trough texture contrast.
    period : tuple[float, float]
        Horizontal and vertical periods.
    phase : tuple[float, float]
        Horizontal and vertical phases in radians.
    """

    mean: float = 0.5
    contrast: float = 0.4
    period: tuple[float, float] = (5.0, 5.0)
    phase: tuple[float, float] = (0.0, 0.0)

    def evaluate(self, x_coord: np.ndarray, y_coord: np.ndarray) -> np.ndarray:
        """Evaluate the eggbox texture at reference-world coordinates."""
        x_wave = 2.0 * np.pi / self.period[0]
        y_wave = 2.0 * np.pi / self.period[1]
        return (self.mean - self.contrast
                + 0.5 * self.contrast
                * (1.0 + np.cos(x_wave * x_coord + self.phase[0]))
                * (1.0 + np.cos(y_wave * y_coord + self.phase[1])))


def quadrature_points(
    rule: RectRule | GaussRule,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return unit-pixel points and weights for a numerical rule."""
    if isinstance(rule, RectRule):
        points = (np.arange(rule.samples_per_axis) + 0.5) / rule.samples_per_axis
        weights = np.full(rule.samples_per_axis, 1.0 / rule.samples_per_axis)
    else:
        points, weights = np.polynomial.legendre.leggauss(rule.points_per_axis)
        points = 0.5 * (points + 1.0)
        weights = 0.5 * weights
    points_x, points_y = np.meshgrid(points, points)
    return points_x.ravel(), points_y.ravel(), np.outer(weights, weights).ravel()


def quantise_image(image: np.ndarray, bits: int) -> np.ndarray:
    """Convert normalised intensities to unsigned camera codes."""
    if bits < 1 or bits > 16:
        raise ValueError("bits must be in [1, 16].")
    maximum = (1 << bits) - 1
    dtype = np.uint8 if bits <= 8 else np.uint16
    return np.rint(np.clip(image, 0.0, 1.0) * maximum).astype(dtype)


__all__ = [
    "AnalyticRule", "EPxIntMapping", "Eggbox", "GaussianPSF", "GaussRule",
    "PxInt2DOpts", "RectRule", "quadrature_points", "quantise_image",
]
