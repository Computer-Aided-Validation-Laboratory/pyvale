"""Canonical signed residual blocks for objectives and sensitivity audits."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
import numpy.typing as npt

from pyvale.vfm.loadregimes import ResolvedLoadRegimes
from pyvale.vfm.metric import MetricResult


FloatArray = npt.NDArray[np.float64]


@dataclass(slots=True, frozen=True)
class ResidualBlockSpec:
    """Configuration for one metric/support/load-regime residual block.

    ``noise_scale`` is an observation standard deviation. Its elementwise use
    is deliberately described as diagonal whitening; spatial and cross-metric
    covariance are not implied. ``observation_weights`` act within the block
    and are normalised to sum to one. ``block_weight`` controls total block
    influence independently of observation count.
    """

    name: str
    metric_index: int
    load_regime: str
    metric_kind: str
    role: str = "training"
    residual_field: str | None = None
    physical_support: float | None = None
    pixel_support: tuple[int, int] | None = None
    bias: npt.ArrayLike | float = 0.0
    noise_scale: npt.ArrayLike | float = 1.0
    observation_weights: npt.ArrayLike | float = 1.0
    block_weight: float = 1.0

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("Residual block name cannot be empty.")
        if self.metric_index < 0:
            raise ValueError("metric_index must be non-negative.")
        if self.load_regime not in {
            "all",
            "pre_yield",
            "onset",
            "developed",
            "late",
        }:
            raise ValueError(f"Unknown load regime {self.load_regime!r}.")
        if not self.metric_kind or not self.role:
            raise ValueError("metric_kind and role cannot be empty.")
        if not np.isfinite(self.block_weight) or self.block_weight < 0.0:
            raise ValueError("block_weight must be finite and non-negative.")
        if self.physical_support is not None and (
            not np.isfinite(self.physical_support)
            or self.physical_support <= 0.0
        ):
            raise ValueError("physical_support must be finite and positive.")
        if self.pixel_support is not None and any(
            value < 3 or value % 2 == 0 for value in self.pixel_support
        ):
            raise ValueError("pixel_support dimensions must be odd and at least 3.")


@dataclass(slots=True, frozen=True)
class PreparedResidualBlock:
    """Frozen observation mask and diagonal transform for one block."""

    spec: ResidualBlockSpec
    source_shape: tuple[int, ...]
    frame_indices: tuple[int, ...]
    valid_indices: npt.NDArray[np.int64]
    bias: FloatArray
    noise_scale: FloatArray
    square_root_weights: FloatArray
    total_observation_count: int

    @property
    def observation_count(self) -> int:
        return int(self.valid_indices.size)

    @property
    def coverage_fraction(self) -> float:
        return self.observation_count / self.total_observation_count

    def diagnostics(self) -> dict[str, object]:
        return {
            "name": self.spec.name,
            "metric_index": self.spec.metric_index,
            "metric_kind": self.spec.metric_kind,
            "load_regime": self.spec.load_regime,
            "frame_indices": list(self.frame_indices),
            "role": self.spec.role,
            "residual_field": self.spec.residual_field,
            "physical_support": self.spec.physical_support,
            "pixel_support": (
                None
                if self.spec.pixel_support is None
                else list(self.spec.pixel_support)
            ),
            "observation_count": self.observation_count,
            "total_observation_count": self.total_observation_count,
            "coverage_fraction": self.coverage_fraction,
            "block_weight": self.spec.block_weight,
            "whitening": "diagonal_standard_deviation",
        }


@dataclass(slots=True, frozen=True)
class CanonicalResidualVector:
    """Canonical vectors plus stable slices back to semantic blocks."""

    signed: FloatArray
    whitened: FloatArray
    weighted: FloatArray
    block_slices: tuple[tuple[str, int, int], ...]

    def block_slice(self, name: str) -> slice:
        for block_name, start, stop in self.block_slices:
            if block_name == name:
                return slice(start, stop)
        raise KeyError(f"Unknown residual block {name!r}.")


@dataclass(slots=True, frozen=True)
class CanonicalResidualLayout:
    """Frozen block ordering, masks, whitening, and aggregation weights."""

    blocks: tuple[PreparedResidualBlock, ...]

    def evaluate(
        self,
        metric_results: Sequence[MetricResult],
    ) -> CanonicalResidualVector:
        signed_parts: list[FloatArray] = []
        whitened_parts: list[FloatArray] = []
        weighted_parts: list[FloatArray] = []
        slices: list[tuple[str, int, int]] = []
        start = 0
        for block in self.blocks:
            source = _metric_residual(
                metric_results,
                block.spec.metric_index,
                block.spec.residual_field,
            )
            if source.shape != block.source_shape:
                raise ValueError(
                    f"Residual block {block.spec.name!r} changed shape from "
                    f"{block.source_shape} to {source.shape}."
                )
            selected = source[np.asarray(block.frame_indices)].ravel()
            values = selected[block.valid_indices]
            if np.any(~np.isfinite(values)):
                raise ValueError(
                    f"Residual block {block.spec.name!r} became non-finite "
                    "inside its frozen observation mask."
                )
            signed = values - block.bias
            whitened = signed / block.noise_scale
            weighted = whitened * block.square_root_weights
            signed_parts.append(signed)
            whitened_parts.append(whitened)
            weighted_parts.append(weighted)
            stop = start + signed.size
            slices.append((block.spec.name, start, stop))
            start = stop
        return CanonicalResidualVector(
            signed=_concatenate(signed_parts),
            whitened=_concatenate(whitened_parts),
            weighted=_concatenate(weighted_parts),
            block_slices=tuple(slices),
        )

    def diagnostics(self) -> dict[str, object]:
        return {
            "observation_count": sum(
                block.observation_count for block in self.blocks
            ),
            "block_count": len(self.blocks),
            "blocks": [block.diagnostics() for block in self.blocks],
        }


def prepare_canonical_residual_layout(
    metric_results: Sequence[MetricResult],
    load_regimes: ResolvedLoadRegimes,
    specs: Sequence[ResidualBlockSpec],
) -> CanonicalResidualLayout:
    """Freeze a canonical residual layout from an accepted reference state."""

    if not specs:
        raise ValueError("At least one residual block specification is required.")
    names = [spec.name for spec in specs]
    if len(set(names)) != len(names):
        raise ValueError("Residual block names must be unique.")

    blocks: list[PreparedResidualBlock] = []
    for spec in specs:
        source = _metric_residual(
            metric_results,
            spec.metric_index,
            spec.residual_field,
        )
        if source.ndim == 0:
            raise ValueError(
                f"Residual block {spec.name!r} must have a load-step axis."
            )
        frame_indices = (
            tuple(range(source.shape[0]))
            if spec.load_regime == "all"
            else load_regimes.indices(spec.load_regime)
        )
        if not frame_indices:
            raise ValueError(f"Residual block {spec.name!r} has no frames.")
        if min(frame_indices) < 0 or max(frame_indices) >= source.shape[0]:
            raise ValueError(
                f"Residual block {spec.name!r} contains out-of-range frames."
            )
        selected = source[np.asarray(frame_indices)]
        bias = _broadcast_block_value(spec.bias, source.shape, frame_indices)
        noise = _broadcast_block_value(
            spec.noise_scale,
            source.shape,
            frame_indices,
        )
        weights = _broadcast_block_value(
            spec.observation_weights,
            source.shape,
            frame_indices,
        )
        valid = (
            np.isfinite(selected)
            & np.isfinite(bias)
            & np.isfinite(noise)
            & (noise > 0.0)
            & np.isfinite(weights)
            & (weights > 0.0)
        ).ravel()
        if not np.any(valid):
            raise ValueError(
                f"Residual block {spec.name!r} has no valid observations."
            )
        valid_indices = np.flatnonzero(valid)
        valid_weights = weights.ravel()[valid_indices]
        valid_weights = valid_weights / np.sum(valid_weights)
        square_root_weights = np.sqrt(spec.block_weight * valid_weights)
        frozen_indices = valid_indices.copy()
        frozen_bias = bias.ravel()[valid_indices].copy()
        frozen_noise = noise.ravel()[valid_indices].copy()
        frozen_weights = square_root_weights.copy()
        for values in (
            frozen_indices,
            frozen_bias,
            frozen_noise,
            frozen_weights,
        ):
            values.setflags(write=False)
        blocks.append(
            PreparedResidualBlock(
                spec=spec,
                source_shape=source.shape,
                frame_indices=tuple(int(index) for index in frame_indices),
                valid_indices=frozen_indices,
                bias=frozen_bias,
                noise_scale=frozen_noise,
                square_root_weights=frozen_weights,
                total_observation_count=int(selected.size),
            )
        )
    return CanonicalResidualLayout(tuple(blocks))


def _metric_residual(
    metric_results: Sequence[MetricResult],
    metric_index: int,
    residual_field: str | None,
) -> FloatArray:
    if metric_index >= len(metric_results):
        raise ValueError(
            f"Metric index {metric_index} is unavailable in "
            f"{len(metric_results)} results."
        )
    result = metric_results[metric_index]
    if residual_field is None:
        values = result.residual
    else:
        fields = result.additional_fields or {}
        if residual_field not in fields:
            raise ValueError(
                f"Metric {metric_index} has no residual field "
                f"{residual_field!r}."
            )
        values = fields[residual_field]
    if values is None:
        raise ValueError(f"Metric {metric_index} has no residual values.")
    return np.asarray(values, dtype=np.float64)


def _broadcast_block_value(
    value: npt.ArrayLike | float,
    source_shape: tuple[int, ...],
    frame_indices: tuple[int, ...],
) -> FloatArray:
    array = np.asarray(value, dtype=np.float64)
    selected_shape = (len(frame_indices), *source_shape[1:])
    if array.ndim == 1 and array.size in {source_shape[0], len(frame_indices)}:
        if array.size == source_shape[0]:
            array = array[np.asarray(frame_indices)]
        array = array.reshape((len(frame_indices),) + (1,) * (len(source_shape) - 1))
    try:
        return np.broadcast_to(array, selected_shape).copy()
    except ValueError:
        try:
            full = np.broadcast_to(array, source_shape)
        except ValueError as exc:
            raise ValueError(
                f"Block transform shape {array.shape} cannot broadcast to "
                f"selected shape {selected_shape} or source shape {source_shape}."
            ) from exc
        return np.asarray(full[np.asarray(frame_indices)], dtype=np.float64).copy()


def _concatenate(parts: list[FloatArray]) -> FloatArray:
    if not parts:
        return np.empty(0, dtype=np.float64)
    return np.concatenate(parts)
