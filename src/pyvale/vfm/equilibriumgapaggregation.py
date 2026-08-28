from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

from pyvale.vfm.metric import MetricResult


@dataclass(slots=True, frozen=True)
class EquilibriumGapAggregationResult:
    """Combined multiscale EGI map and scalar diagnostic.

    This result contains candidate EGI fields. The baseline values are fixed
    scalar values, normally computed earlier from a reference stress field.

    Attributes
    ----------
    combined_baseline_scaled_egi_map : npt.NDArray[np.float64]
        The combined 2D map after baseline scaling and window weighting.
    combined_egi_spatial_rms : float
        Spatial RMS of ``combined_baseline_scaled_egi_map``.
    baseline_scaled_egi_temporal_rms_maps : tuple[npt.NDArray[np.float64], ...]
        Candidate EGI temporal RMS maps divided by their baseline values.
    egi_baseline_values : npt.NDArray[np.float64]
        Positive scalar baseline values, one per EGI window size.
    window_weights : npt.NDArray[np.float64]
        Normalised weights used to combine the window sizes.
    spatial_weights : tuple[npt.NDArray[np.float64], ...] | None
        Optional centre weights applied as ``sqrt(weight)`` to each map.
    """

    combined_baseline_scaled_egi_map: npt.NDArray[np.float64]
    combined_egi_spatial_rms: float
    baseline_scaled_egi_temporal_rms_maps: tuple[npt.NDArray[np.float64], ...]
    egi_baseline_values: npt.NDArray[np.float64]
    window_weights: npt.NDArray[np.float64]
    spatial_weights: tuple[npt.NDArray[np.float64], ...] | None


def aggregate_equilibrium_gap_results(
    metric_results: Sequence[MetricResult],
    *,
    egi_baseline_values: Sequence[float] | npt.NDArray[np.float64] | None = None,
    window_weights: Sequence[float] | npt.NDArray[np.float64] | None = None,
    spatial_weights: Sequence[npt.NDArray[np.float64]] | None = None,
) -> EquilibriumGapAggregationResult:
    """Combine one or more single-window EGI results into a map and scalar.

    This module does not compute EGI from stress. It combines already computed
    EGI ``MetricResult`` objects, where each result usually corresponds to one
    virtual-window size.

    For each window size ``k``:

    ``baseline_scaled_egi_temporal_rms_map_k =
    egi_temporal_rms_map_k / egi_baseline_value_k``

    These baseline-scaled maps are then combined with ``window_weights``. NaN
    values are ignored per spatial location and the available weights are
    renormalised there, so a point can still be used when only some window
    sizes are valid. The scalar value is always the spatial RMS of the
    combined map.
    """

    if not metric_results:
        raise ValueError("At least one equilibrium-gap metric result is required.")

    # These are candidate EGI maps, not maps computed from the reference stress.
    egi_temporal_rms_maps = tuple(
        extract_equilibrium_gap_temporal_rms(metric_result)
        for metric_result in metric_results
    )
    _validate_matching_map_shapes(egi_temporal_rms_maps)

    # Baseline values are fixed scalars, one per window size. They should be
    # computed outside this function from a reference stress field.
    resolved_egi_baseline_values = _resolve_egi_baseline_values(
        egi_baseline_values,
        len(egi_temporal_rms_maps),
    )

    # Window weights are used to combine the baseline-scaled maps. They should be
    # computed outside this function, e.g. from the window areas.
    resolved_window_weights = _resolve_window_weights(
        window_weights,
        len(egi_temporal_rms_maps),
    )
    resolved_spatial_weights = _resolve_spatial_weights(
        spatial_weights,
        egi_temporal_rms_maps,
    )

    # Scale each EGI temporal RMS map by its baseline value.
    baseline_scaled_egi_temporal_rms_maps = []
    for egi_temporal_rms_map, egi_baseline_value in zip(
        egi_temporal_rms_maps,
        resolved_egi_baseline_values,
        strict=True,
    ):
        if not np.isfinite(egi_baseline_value) or egi_baseline_value <= 0.0:
            raise ValueError(
                f"EGI baseline value {egi_baseline_value} must be finite and greater than zero."
            )

        # Divide EGI temporal RMS map pointwise by its baseline value (NaN values are preserved)
        baseline_scaled_egi_temporal_rms_maps.append(
            egi_temporal_rms_map / egi_baseline_value
        )
    # Convert to tuple for immutability and consistency with the dataclass
    baseline_scaled_egi_temporal_rms_maps = tuple(baseline_scaled_egi_temporal_rms_maps)
    
    # Initalise array to accumulate the weighted sum 
    combined_sum = np.zeros_like(
        baseline_scaled_egi_temporal_rms_maps[0],
        dtype=np.float64,
    )
    # Initalise array to accumulate the sum of valid weights
    valid_weight_sum = np.zeros_like(
        baseline_scaled_egi_temporal_rms_maps[0],
        dtype=np.float64,
    )
    # Loop over each baseline-scaled EGI temporal RMS map and its corresponding window weight
    for map_index, (
        baseline_scaled_egi_temporal_rms_map,
        window_weight,
    ) in enumerate(
        zip(
            baseline_scaled_egi_temporal_rms_maps,
            resolved_window_weights,
            strict=True,
        )
    ):
        if resolved_spatial_weights is not None:
            baseline_scaled_egi_temporal_rms_map = (
                baseline_scaled_egi_temporal_rms_map
                * np.sqrt(resolved_spatial_weights[map_index])
            )
        # Identify valid (finite) points in the current baseline-scaled EGI temporal RMS map
        valid = np.isfinite(baseline_scaled_egi_temporal_rms_map)
        # Accumulate the weighted sum for valid points
        combined_sum[valid] = (
            combined_sum[valid]
            + window_weight * baseline_scaled_egi_temporal_rms_map[valid]
        )
        # Accumulate the sum of valid weights for valid points
        valid_weight_sum[valid] = valid_weight_sum[valid] + window_weight

    # Initialise the combined baseline-scaled EGI map with NaN values
    combined_baseline_scaled_egi_map = np.full_like(
        combined_sum,
        np.nan,
        dtype=np.float64,
    )

    # Identify points where the sum of valid weights is greater than zero
    # and compute the combined baseline-scaled EGI map for those points
    valid_combination = valid_weight_sum > 0.0
    combined_baseline_scaled_egi_map[valid_combination] = (
        combined_sum[valid_combination]
        / valid_weight_sum[valid_combination]
    )

    return EquilibriumGapAggregationResult(
        combined_baseline_scaled_egi_map=combined_baseline_scaled_egi_map,
        combined_egi_spatial_rms=calculate_nan_rms(
            combined_baseline_scaled_egi_map,
        ),
        baseline_scaled_egi_temporal_rms_maps=(
            baseline_scaled_egi_temporal_rms_maps
        ),
        egi_baseline_values=resolved_egi_baseline_values,
        window_weights=resolved_window_weights,
        spatial_weights=resolved_spatial_weights,
    )


def combine_equilibrium_gap_maps(
    metric_results: Sequence[MetricResult],
    *,
    egi_baseline_values: Sequence[float] | npt.NDArray[np.float64] | None = None,
    window_weights: Sequence[float] | npt.NDArray[np.float64] | None = None,
    spatial_weights: Sequence[npt.NDArray[np.float64]] | None = None,
) -> npt.NDArray[np.float64]:
    """Return the baseline-scaled, window-weighted combined EGI map."""

    return aggregate_equilibrium_gap_results(
        metric_results,
        egi_baseline_values=egi_baseline_values,
        window_weights=window_weights,
        spatial_weights=spatial_weights,
    ).combined_baseline_scaled_egi_map


def calculate_combined_equilibrium_gap_spatial_rms(
    metric_results: Sequence[MetricResult],
    *,
    egi_baseline_values: Sequence[float] | npt.NDArray[np.float64] | None = None,
    window_weights: Sequence[float] | npt.NDArray[np.float64] | None = None,
    spatial_weights: Sequence[npt.NDArray[np.float64]] | None = None,
) -> float:
    """Return the spatial RMS of the combined EGI map."""

    return aggregate_equilibrium_gap_results(
        metric_results,
        egi_baseline_values=egi_baseline_values,
        window_weights=window_weights,
        spatial_weights=spatial_weights,
    ).combined_egi_spatial_rms


def extract_equilibrium_gap_temporal_rms(
    metric_result: MetricResult,
) -> npt.NDArray[np.float64]:
    """Return an EGI temporal RMS map, computing it if the metric omitted it."""

    metadata = metric_result.additional_fields or {}
    weighted_temporal_rms = metadata.get("weighted_temporal_rms")

    # If the metric result already contains a weighted temporal RMS map, return it.
    if weighted_temporal_rms is not None:
        return np.asarray(weighted_temporal_rms, dtype=np.float64)

    # Otherwise, compute the weighted temporal RMS from the normalised gap and temporal weights.
    normalised_gap = metadata.get("normalised_gap", metric_result.residual)
    if normalised_gap is None:
        raise ValueError("Equilibrium-gap metric result does not contain a normalised gap.")

    temporal_weights = metadata.get("temporal_weights")
    if temporal_weights is None:
        raise ValueError("Equilibrium-gap metric result does not contain temporal weights.")

    return calculate_weighted_temporal_rms(
        np.asarray(normalised_gap, dtype=np.float64),
        np.asarray(temporal_weights, dtype=np.float64),
    )


def calculate_weighted_temporal_rms(
    values: npt.NDArray[np.float64],
    temporal_weights: npt.NDArray[np.float64],
) -> npt.NDArray[np.float64]:
    """Return the weighted RMS over the temporal axis.

    ``values`` is expected to have time/load step as its first dimension, with
    any number of trailing spatial or component dimensions. ``temporal_weights``
    must contain one weight per time step.

    The result has the same shape as ``values`` without the leading temporal
    dimension. NaNs are ignored at each output location; locations with no
    finite weighted values are returned as NaN.
    """

    resolved_values = np.asarray(values, dtype=np.float64)
    resolved_weights = np.asarray(temporal_weights, dtype=np.float64)
    if resolved_values.ndim < 1:
        raise ValueError("Temporal RMS values must have at least one dimension.")
    if resolved_values.shape[0] != resolved_weights.shape[0]:
        raise ValueError(
            "Temporal weights must match the first EGI dimension: "
            f"{resolved_weights.shape[0]} vs {resolved_values.shape[0]}."
        )

    # Reshape one-dimensional temporal weights so they broadcast across every
    # non-temporal axis of the values array.
    weight_shape = resolved_weights.shape + (1,) * (resolved_values.ndim - 1)
    weighted_squared = resolved_values**2 * resolved_weights.reshape(weight_shape)

    # Count finite weighted samples per output location, then divide by that
    # local count so missing timesteps do not make the RMS NaN.
    valid_counts = np.sum(np.isfinite(weighted_squared), axis=0)
    weighted_sum = np.nansum(weighted_squared, axis=0)

    temporal_rms = np.full(valid_counts.shape, np.nan, dtype=np.float64)
    valid = valid_counts > 0

    # Compute the RMS only for locations with at least one valid weighted sample.
    temporal_rms[valid] = np.sqrt(weighted_sum[valid] / valid_counts[valid])
    return temporal_rms


def calculate_nan_rms(
    values: npt.NDArray[np.float64],
) -> float:
    """Return RMS over finite values, ignoring NaN-masked points."""

    finite_values = np.asarray(values, dtype=np.float64)
    finite_values = finite_values[np.isfinite(finite_values)]
    if finite_values.size == 0:
        return float("nan")
    return float(np.sqrt(np.mean(finite_values**2)))


def infer_window_area_weights(
    metric_results: Sequence[MetricResult],
) -> npt.NDArray[np.float64]:
    """Infer normalised window-area weights from EGI ``window_size`` metadata.
    
    The area of each window is computed as the product of its two dimensions, and
    the weights are normalised to sum to one. 
    """

    if not metric_results:
        raise ValueError("At least one equilibrium-gap metric result is required.")

    areas = []
    for metric_result in metric_results:
        metadata = metric_result.additional_fields or {}
        window_size = metadata.get("window_size")
        if window_size is None:
            raise ValueError(
                "Equilibrium-gap metric result does not contain window_size metadata."
            )
        resolved_window_size = np.asarray(window_size, dtype=np.float64)
        if resolved_window_size.shape != (2,) or np.any(resolved_window_size <= 0.0):
            raise ValueError(
                f"Invalid EGI window_size metadata {resolved_window_size}."
            )
        areas.append(float(np.prod(resolved_window_size)))

    return _normalise_nonnegative_weights(np.asarray(areas, dtype=np.float64))


def _validate_matching_map_shapes(
    maps: Sequence[npt.NDArray[np.float64]],
) -> None:
    expected_shape = maps[0].shape
    for map_index, values in enumerate(maps[1:], start=1):
        if values.shape != expected_shape:
            raise ValueError(
                "All EGI temporal RMS maps must have the same shape: "
                f"{expected_shape} vs map {map_index} shape {values.shape}."
            )


def _resolve_egi_baseline_values(
    egi_baseline_values: Sequence[float] | npt.NDArray[np.float64] | None,
    count: int,
) -> npt.NDArray[np.float64]:

    if egi_baseline_values is None:
        raise ValueError("EGI baseline values are required.")
    
    # OLD: If no baseline values are provided, return an array of ones. 
    # if egi_baseline_values is None:
    #     return np.ones(count, dtype=np.float64)

    resolved = np.asarray(egi_baseline_values, dtype=np.float64)
    if resolved.shape != (count,):
        raise ValueError(
            "egi_baseline_values must contain one value per EGI metric result: "
            f"{resolved.shape} vs ({count},)."
        )
    if np.any(~np.isfinite(resolved)) or np.any(resolved <= 0.0):
        raise ValueError("egi_baseline_values must be finite and greater than zero.")
    return resolved


def _resolve_window_weights(
    window_weights: Sequence[float] | npt.NDArray[np.float64] | None,
    count: int,
) -> npt.NDArray[np.float64]:

    if window_weights is None:
        raise ValueError("Window weights are required.")
    
    # OLD: if window_weights is None:
    #     return np.full(count, 1.0 / count, dtype=np.float64)

    resolved = np.asarray(window_weights, dtype=np.float64)
    if resolved.shape != (count,):
        raise ValueError(
            "window_weights must contain one value per EGI metric result: "
            f"{resolved.shape} vs ({count},)."
        )
    return _normalise_nonnegative_weights(resolved)


def _resolve_spatial_weights(
    spatial_weights: Sequence[npt.NDArray[np.float64]] | None,
    maps: Sequence[npt.NDArray[np.float64]],
) -> tuple[npt.NDArray[np.float64], ...] | None:
    if spatial_weights is None:
        return None
    if len(spatial_weights) != len(maps):
        raise ValueError(
            "spatial_weights must contain one map per EGI metric result: "
            f"{len(spatial_weights)} vs {len(maps)}."
        )

    resolved: list[npt.NDArray[np.float64]] = []
    for map_index, (weights, values) in enumerate(
        zip(spatial_weights, maps, strict=True)
    ):
        array = np.asarray(weights, dtype=np.float64)
        if array.shape != values.shape:
            raise ValueError(
                f"EGI spatial weight map {map_index} has shape {array.shape}; "
                f"expected {values.shape}."
            )
        valid = np.isfinite(values)
        if np.any(~np.isfinite(array[valid])) or np.any(array[valid] < 0.0):
            raise ValueError(
                "EGI spatial weights must be finite and non-negative on "
                "valid EGI centres."
            )
        if not np.any(array[valid] > 0.0):
            raise ValueError(
                f"EGI spatial weight map {map_index} has no positive valid weight."
            )
        resolved.append(array.copy())
    return tuple(resolved)


def _normalise_nonnegative_weights(
    weights: npt.NDArray[np.float64],
) -> npt.NDArray[np.float64]:
    if np.any(~np.isfinite(weights)) or np.any(weights < 0.0):
        raise ValueError("Weights must be finite and non-negative.")
    total = float(np.sum(weights))
    if total <= 0.0:
        raise ValueError("At least one weight must be greater than zero.")
    return weights / total
