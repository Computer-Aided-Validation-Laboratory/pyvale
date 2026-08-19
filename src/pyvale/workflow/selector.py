"""Pixel-coordinate selectors for grid-valued workflow metrics."""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum

import numpy as np


class ISpatialSelector(ABC):
    """Select values from a result grid in pixel coordinates."""

    @abstractmethod
    def select(self, values: np.ndarray, pixels_x: np.ndarray,
               pixels_y: np.ndarray) -> np.ndarray:
        """Return selected values."""


class ESpatialReduction(Enum):
    """Supported reductions for selected field values."""

    SIGNED_MEAN = "signed_mean"


class EStrainComponent(Enum):
    """In-plane strain components used by convergence-study functions."""

    EPS_XX = "eps_xx"
    EPS_XY = "eps_xy"
    EPS_YX = "eps_yx"
    EPS_YY = "eps_yy"


@dataclass(frozen=True, slots=True)
class SignalExtraction:
    """Describe one spatial extraction from a requested strain component."""

    component: EStrainComponent
    selector: ISpatialSelector
    reduction: ESpatialReduction = ESpatialReduction.SIGNED_MEAN

    def reduce(
        self,
        values: np.ndarray,
        pixels_x: np.ndarray,
        pixels_y: np.ndarray,
    ) -> float:
        """Select finite values and calculate the configured reduction."""
        selected = self.selector.select(values, pixels_x, pixels_y)
        selected = np.asarray(selected, dtype=np.float64)
        selected = selected[np.isfinite(selected)]
        if selected.size == 0:
            return float("nan")
        if self.reduction is ESpatialReduction.SIGNED_MEAN:
            return float(np.mean(selected))
        raise ValueError(f"Unsupported spatial reduction: {self.reduction}.")


@dataclass(frozen=True, slots=True)
class PointSelector(ISpatialSelector):
    """Select the nearest result-grid value to one pixel position."""

    position_px: np.ndarray

    def select(self, values: np.ndarray, pixels_x: np.ndarray,
               pixels_y: np.ndarray) -> np.ndarray:
        """Return the nearest grid value."""
        distance = (
            (pixels_x - self.position_px[0]) ** 2
            + (pixels_y - self.position_px[1]) ** 2
        )
        return np.asarray((values.flat[np.nanargmin(distance)],))


@dataclass(frozen=True, slots=True)
class LineSelector(ISpatialSelector):
    """Select grid values within a pixel-width line band."""

    start_px: np.ndarray
    end_px: np.ndarray
    width_px: float = 0.0

    def select(self, values: np.ndarray, pixels_x: np.ndarray,
               pixels_y: np.ndarray) -> np.ndarray:
        """Return finite values inside the line band."""
        direction = self.end_px - self.start_px
        length = np.linalg.norm(direction)
        if length == 0.0:
            return PointSelector(self.start_px).select(values, pixels_x, pixels_y)
        distance = np.abs(direction[0] * (self.start_px[1] - pixels_y)
                          - (self.start_px[0] - pixels_x) * direction[1]) / length
        projection = ((pixels_x - self.start_px[0]) * direction[0]
                      + (pixels_y - self.start_px[1]) * direction[1]) / length
        mask = (
            (distance <= self.width_px / 2.0)
            & (projection >= 0.0)
            & (projection <= length)
        )
        return values[mask]


@dataclass(frozen=True, slots=True)
class AreaSelector(ISpatialSelector):
    """Select a rectangular pixel-coordinate area."""

    lower_px: np.ndarray
    upper_px: np.ndarray

    def select(self, values: np.ndarray, pixels_x: np.ndarray,
               pixels_y: np.ndarray) -> np.ndarray:
        """Return values in the inclusive rectangular area."""
        mask = ((pixels_x >= self.lower_px[0]) & (pixels_x <= self.upper_px[0])
                & (pixels_y >= self.lower_px[1]) & (pixels_y <= self.upper_px[1]))
        return values[mask]


class FullFieldSelector(ISpatialSelector):
    """Select every grid value."""

    def select(self, values: np.ndarray, pixels_x: np.ndarray,
               pixels_y: np.ndarray) -> np.ndarray:
        """Return all grid values."""
        return values.ravel()


@dataclass(frozen=True, slots=True)
class MaskSelector(ISpatialSelector):
    """Select values through a Boolean mask aligned with the result grid."""

    mask: np.ndarray

    def select(self, values: np.ndarray, pixels_x: np.ndarray,
               pixels_y: np.ndarray) -> np.ndarray:
        """Return values selected by the stored Boolean mask."""
        if self.mask.shape != values.shape:
            raise ValueError("MaskSelector mask must match result value shape.")
        return values[self.mask]
