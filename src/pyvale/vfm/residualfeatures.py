"""Reusable reductions for spatiotemporal VFM residual fields.

The functions in this module deliberately know nothing about a particular
specimen or constitutive law.  They turn signed residual arrays into robust,
well-diagnosed scalar features for objective functions and offline screens.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Literal

import numpy as np
import numpy.typing as npt
from scipy.ndimage import gaussian_filter


FloatArray = npt.NDArray[np.float64]


@dataclass(slots=True, frozen=True)
class ResidualFeatureResult:
    """A scalar residual feature and its effective support."""

    value: float
    valid_count: int
    effective_count: float
    weight_sum: float

    def diagnostics(self) -> dict[str, float | int]:
        return asdict(self)


def normalised_weights(
    values: npt.ArrayLike,
    weights: npt.ArrayLike | None = None,
) -> tuple[FloatArray, FloatArray]:
    """Return finite values and non-negative weights normalised to sum to one."""

    resolved_values = np.asarray(values, dtype=np.float64)
    if weights is None:
        resolved_weights = np.ones(resolved_values.shape, dtype=np.float64)
    else:
        resolved_weights = np.broadcast_to(
            np.asarray(weights, dtype=np.float64), resolved_values.shape
        ).copy()
    valid = (
        np.isfinite(resolved_values)
        & np.isfinite(resolved_weights)
        & (resolved_weights > 0.0)
    )
    flat_values = resolved_values[valid]
    flat_weights = resolved_weights[valid]
    if flat_values.size == 0:
        raise ValueError("Residual feature has no finite, positively weighted values.")
    flat_weights /= np.sum(flat_weights)
    return flat_values, flat_weights


def weighted_rms(
    values: npt.ArrayLike,
    weights: npt.ArrayLike | None = None,
) -> ResidualFeatureResult:
    """Root weighted mean square over finite observations."""

    samples, probabilities = normalised_weights(values, weights)
    value = float(np.sqrt(np.sum(probabilities * samples**2)))
    return _result(value, probabilities)


def weighted_cvar_abs(
    values: npt.ArrayLike,
    weights: npt.ArrayLike | None = None,
    *,
    quantile: float = 0.90,
) -> ResidualFeatureResult:
    """Weighted conditional mean of ``abs(values)`` above a quantile.

    Fractional weight at the threshold is retained so the selected tail has
    exactly ``1 - quantile`` probability.  This makes the result stable when
    observations have unequal weights.
    """

    if not 0.0 <= quantile < 1.0:
        raise ValueError("quantile must satisfy 0 <= quantile < 1.")
    samples, probabilities = normalised_weights(values, weights)
    magnitudes = np.abs(samples)
    order = np.argsort(magnitudes)[::-1]
    magnitudes = magnitudes[order]
    probabilities = probabilities[order]
    tail_mass = 1.0 - quantile
    selected = np.minimum(probabilities, np.maximum(tail_mass - np.cumsum(probabilities) + probabilities, 0.0))
    value = float(np.sum(selected * magnitudes) / tail_mass)
    return _result(value, selected[selected > 0.0] / tail_mass)


def coherent_rms(
    values: npt.ArrayLike,
    weights: npt.ArrayLike | None = None,
    *,
    sigma_pixels: float | tuple[float, ...],
    spatial_axes: tuple[int, ...] = (-2, -1),
    truncate: float = 4.0,
) -> ResidualFeatureResult:
    """RMS after mask-normalised Gaussian smoothing of a signed field."""

    field = np.asarray(values, dtype=np.float64)
    if field.ndim < len(spatial_axes):
        raise ValueError("spatial_axes exceed residual dimensionality.")
    axes = tuple(axis % field.ndim for axis in spatial_axes)
    if len(set(axes)) != len(axes):
        raise ValueError("spatial_axes must be unique.")
    sigma_values = (
        (float(sigma_pixels),) * len(axes)
        if np.ndim(sigma_pixels) == 0
        else tuple(float(value) for value in sigma_pixels)
    )
    if len(sigma_values) != len(axes) or any(value < 0.0 for value in sigma_values):
        raise ValueError("sigma_pixels must provide one non-negative value per spatial axis.")
    sigma = [0.0] * field.ndim
    for axis, value in zip(axes, sigma_values, strict=True):
        sigma[axis] = value

    if weights is None:
        raw_weights = np.ones(field.shape, dtype=np.float64)
    else:
        raw_weights = np.broadcast_to(np.asarray(weights, dtype=np.float64), field.shape).copy()
    valid = np.isfinite(field) & np.isfinite(raw_weights) & (raw_weights > 0.0)
    numerator = gaussian_filter(
        np.where(valid, field * raw_weights, 0.0), sigma=sigma, truncate=truncate
    )
    denominator = gaussian_filter(
        np.where(valid, raw_weights, 0.0), sigma=sigma, truncate=truncate
    )
    smoothed = np.full(field.shape, np.nan, dtype=np.float64)
    supported = valid & (denominator > np.finfo(np.float64).eps)
    smoothed[supported] = numerator[supported] / denominator[supported]
    return weighted_rms(smoothed, np.where(supported, raw_weights, 0.0))


def projected_rms(
    values: npt.ArrayLike,
    basis: npt.ArrayLike,
    weights: npt.ArrayLike | None = None,
) -> ResidualFeatureResult:
    """RMS energy of a residual projected onto an orthonormal column basis."""

    samples, probabilities = normalised_weights(values, weights)
    resolved_basis = np.asarray(basis, dtype=np.float64)
    if resolved_basis.ndim != 2 or resolved_basis.shape[0] != samples.size:
        raise ValueError(
            "basis must have one row per finite weighted residual observation."
        )
    if resolved_basis.shape[1] == 0:
        return _result(0.0, probabilities)
    whitened = np.sqrt(probabilities) * samples
    weighted_basis = np.sqrt(probabilities)[:, np.newaxis] * resolved_basis
    orthonormal, _ = np.linalg.qr(weighted_basis, mode="reduced")
    value = float(np.linalg.norm(orthonormal.T @ whitened))
    return _result(value, probabilities)


def physical_length_to_odd_pixels(length: float, spacing: float, *, minimum: int = 3) -> int:
    """Convert a physical support length to the nearest valid odd pixel count."""

    if not np.isfinite(length) or length <= 0.0:
        raise ValueError("length must be finite and positive.")
    if not np.isfinite(spacing) or spacing <= 0.0:
        raise ValueError("spacing must be finite and positive.")
    pixels = max(int(minimum), int(np.rint(length / spacing)))
    if pixels % 2 == 0:
        lower = pixels - 1
        upper = pixels + 1
        pixels = lower if abs(lower * spacing - length) <= abs(upper * spacing - length) else upper
    return max(pixels, minimum if minimum % 2 else minimum + 1)


def smooth_positive_part(value: float | FloatArray, *, temperature: float) -> float | FloatArray:
    """Numerically stable softplus approximation to ``max(value, 0)``."""

    if temperature <= 0.0 or not np.isfinite(temperature):
        raise ValueError("temperature must be finite and positive.")
    resolved = np.asarray(value, dtype=np.float64) / temperature
    output = temperature * np.logaddexp(0.0, resolved)
    return float(output) if output.ndim == 0 else output


def _result(value: float, probabilities: FloatArray) -> ResidualFeatureResult:
    positive = probabilities[probabilities > 0.0]
    effective = float(1.0 / np.sum(positive**2)) if positive.size else 0.0
    return ResidualFeatureResult(
        value=float(value),
        valid_count=int(positive.size),
        effective_count=effective,
        weight_sum=float(np.sum(positive)),
    )
