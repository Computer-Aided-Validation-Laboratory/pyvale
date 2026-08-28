from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

from pyvale.vfm.equilibriumgapaggregation import (
    EquilibriumGapAggregationResult,
    aggregate_equilibrium_gap_results,
)
from pyvale.vfm.metric import MetricResult
from pyvale.vfm.objectivefunc import IScalarObjectiveFunction


@dataclass(slots=True, frozen=True)
class ForceAndEquilibriumGapObjectiveResult:
    """Diagnostics from the mixed FRE plus EGI scalar objective."""

    total_cost: float
    force_cost: float
    equilibrium_gap_cost: float
    equilibrium_gap_aggregation: EquilibriumGapAggregationResult | None


class ScalarForceAndEquilibriumGapObjective(IScalarObjectiveFunction):
    """Scalar objective combining force reconstruction error and EGI.

    The optimiser still receives a single scalar. Internally, the objective
    separates force-reconstruction metric results from equilibrium-gap metric
    results, aggregates the EGI windows into one map, and combines that EGI
    scalar with the force-reconstruction scalar using the supplied term
    weights. This differs from ``CombinedForceAndEquilibriumGapObjective``,
    which combines independent scalar EGI values from each window.
    """

    def __init__(
        self,
        *,
        force_weight: float = 1.0,
        equilibrium_gap_weight: float = 1.0,
        egi_baseline_values: Sequence[float] | npt.NDArray[np.float64] | None = None,
        egi_window_weights: Sequence[float] | npt.NDArray[np.float64] | None = None,
        force_metric_weights: Sequence[float] | npt.NDArray[np.float64] | None = None,
        force_result_indices: Sequence[int] | None = None,
        equilibrium_gap_result_indices: Sequence[int] | None = None,
    ) -> None:
        self.force_weight = _validate_term_weight(force_weight, "force_weight")
        self.equilibrium_gap_weight = _validate_term_weight(
            equilibrium_gap_weight,
            "equilibrium_gap_weight",
        )
        self.egi_baseline_values = egi_baseline_values
        self.egi_window_weights = egi_window_weights
        self.force_metric_weights = force_metric_weights
        self.force_result_indices = (
            tuple(force_result_indices)
            if force_result_indices is not None
            else None
        )
        self.equilibrium_gap_result_indices = (
            tuple(equilibrium_gap_result_indices)
            if equilibrium_gap_result_indices is not None
            else None
        )
        self.last_result: ForceAndEquilibriumGapObjectiveResult | None = None

    def evaluate(
        self,
        metric_results: list[MetricResult],
    ) -> float:
        force_results, equilibrium_gap_results = self._split_metric_results(metric_results)

        force_cost = 0.0
        if self.force_weight > 0.0:
            if not force_results:
                raise ValueError("Force objective term is active but no force metric results were found.")
            force_cost = combine_force_reconstruction_errors(
                force_results,
                metric_weights=self.force_metric_weights,
            )

        equilibrium_gap_cost = 0.0
        equilibrium_gap_aggregation = None
        if self.equilibrium_gap_weight > 0.0:
            if not equilibrium_gap_results:
                raise ValueError(
                    "EGI objective term is active but no equilibrium-gap metric results were found."
            )
            equilibrium_gap_aggregation = aggregate_equilibrium_gap_results(
                equilibrium_gap_results,
                egi_baseline_values=self.egi_baseline_values,
                window_weights=self.egi_window_weights,
            )
            equilibrium_gap_cost = (
                equilibrium_gap_aggregation.combined_egi_spatial_rms
            )

        total_cost = (
            self.force_weight * force_cost
            + self.equilibrium_gap_weight * equilibrium_gap_cost
        )
        self.last_result = ForceAndEquilibriumGapObjectiveResult(
            total_cost=float(total_cost),
            force_cost=float(force_cost),
            equilibrium_gap_cost=float(equilibrium_gap_cost),
            equilibrium_gap_aggregation=equilibrium_gap_aggregation,
        )
        return float(total_cost)

    def _split_metric_results(
        self,
        metric_results: Sequence[MetricResult],
    ) -> tuple[list[MetricResult], list[MetricResult]]:
        if (
            self.force_result_indices is None
            and self.equilibrium_gap_result_indices is None
        ):
            force_results = []
            equilibrium_gap_results = []
            for metric_result in metric_results:
                metadata = metric_result.additional_fields or {}
                if _is_equilibrium_gap_result(metadata):
                    equilibrium_gap_results.append(metric_result)
                elif _is_force_reconstruction_result(metadata):
                    force_results.append(metric_result)
            return force_results, equilibrium_gap_results

        selected_indices = set(self.force_result_indices or ()) | set(
            self.equilibrium_gap_result_indices or ()
        )
        remaining_results = [
            metric_result
            for index, metric_result in enumerate(metric_results)
            if index not in selected_indices
        ]

        if self.force_result_indices is not None:
            force_results = _select_metric_results(
                metric_results,
                self.force_result_indices,
            )
        else:
            force_results = [
                metric_result
                for metric_result in remaining_results
                if _is_force_reconstruction_result(metric_result.additional_fields or {})
            ]

        if self.equilibrium_gap_result_indices is not None:
            equilibrium_gap_results = _select_metric_results(
                metric_results,
                self.equilibrium_gap_result_indices,
            )
        else:
            equilibrium_gap_results = [
                metric_result
                for metric_result in remaining_results
                if _is_equilibrium_gap_result(metric_result.additional_fields or {})
            ]
        return force_results, equilibrium_gap_results


def combine_force_reconstruction_errors(
    metric_results: Sequence[MetricResult],
    *,
    metric_weights: Sequence[float] | npt.NDArray[np.float64] | None = None,
) -> float:
    """Combine one or more force-reconstruction results into one RMS scalar."""

    if not metric_results:
        raise ValueError("At least one force metric result is required.")

    costs = np.asarray(
        [
            calculate_force_reconstruction_spatiotemporal_rms(metric_result)
            for metric_result in metric_results
        ],
        dtype=np.float64,
    )
    weights = _resolve_metric_weights(metric_weights, len(costs))
    return float(np.sqrt(np.sum(weights * costs**2)))


def calculate_force_reconstruction_spatiotemporal_rms(
    metric_result: MetricResult,
    *,
    spatial_weights: npt.NDArray[np.float64] | None = None,
) -> float:
    """Compute weighted RMS of a force-reconstruction residual."""

    metadata = metric_result.additional_fields or {}
    residual = metadata.get("normalised_residual", metric_result.residual)
    if residual is None:
        raise ValueError("Force metric result does not contain a residual.")

    weighted_residual = np.asarray(residual, dtype=np.float64)
    temporal_weights = metadata.get("temporal_weights")
    if temporal_weights is not None:
        weighted_residual = _apply_axis_weights(
            weighted_residual,
            np.asarray(temporal_weights, dtype=np.float64),
            axis=0,
        )

    resolved_weight_source = (
        metadata.get("spatial_weights")
        if spatial_weights is None
        else spatial_weights
    )
    if resolved_weight_source is not None:
        resolved_spatial_weights = np.asarray(resolved_weight_source, dtype=np.float64)
        if resolved_spatial_weights.ndim == 0:
            weighted_residual = weighted_residual * np.sqrt(float(resolved_spatial_weights))
        else:
            weighted_residual = _apply_axis_weights(
                weighted_residual,
                resolved_spatial_weights,
                axis=weighted_residual.ndim - 1,
            )

    finite_values = weighted_residual[np.isfinite(weighted_residual)]
    if finite_values.size == 0:
        return float("nan")
    return float(np.sqrt(np.sum(finite_values**2)))


def _apply_axis_weights(
    values: npt.NDArray[np.float64],
    weights: npt.NDArray[np.float64],
    *,
    axis: int,
) -> npt.NDArray[np.float64]:
    if values.shape[axis] != weights.shape[0]:
        raise ValueError(
            "Weights do not match residual dimension: "
            f"{weights.shape[0]} vs {values.shape[axis]}."
        )
    weight_shape = [1] * values.ndim
    weight_shape[axis] = weights.shape[0]
    return values * np.sqrt(weights.reshape(tuple(weight_shape)))


def _select_metric_results(
    metric_results: Sequence[MetricResult],
    indices: tuple[int, ...] | None,
) -> list[MetricResult]:
    if indices is None:
        return []
    return [metric_results[index] for index in indices]


def _is_equilibrium_gap_result(
    metadata: dict,
) -> bool:
    return "normalised_gap" in metadata and "window_size" in metadata


def _is_force_reconstruction_result(
    metadata: dict,
) -> bool:
    return "normalised_residual" in metadata and "reconstructed_force" in metadata


def _validate_term_weight(
    weight: float,
    name: str,
) -> float:
    resolved = float(weight)
    if not np.isfinite(resolved) or resolved < 0.0:
        raise ValueError(f"{name} must be finite and non-negative.")
    return resolved


def _resolve_metric_weights(
    metric_weights: Sequence[float] | npt.NDArray[np.float64] | None,
    count: int,
) -> npt.NDArray[np.float64]:
    if metric_weights is None:
        return np.full(count, 1.0 / count, dtype=np.float64)

    resolved = np.asarray(metric_weights, dtype=np.float64)
    if resolved.shape != (count,):
        raise ValueError(
            "metric_weights must contain one value per force metric result: "
            f"{resolved.shape} vs ({count},)."
        )
    if np.any(~np.isfinite(resolved)) or np.any(resolved < 0.0):
        raise ValueError("metric_weights must be finite and non-negative.")
    total = float(np.sum(resolved))
    if total <= 0.0:
        raise ValueError("At least one metric weight must be greater than zero.")
    return resolved / total
