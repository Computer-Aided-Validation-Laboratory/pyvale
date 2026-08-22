"""Combined scalar force-reconstruction error and equilibrium-gap objective.

The formulation is based on the refined identification objective in Hamill
et al. (2026).
"""

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

from pyvale.vfm.metric import MetricResult
from pyvale.vfm.objectivefunc import IScalarObjectiveFunction
from pyvale.vfm.objectivefuncfreandegi import (
    calculate_force_reconstruction_spatiotemporal_rms,
)


@dataclass(slots=True, frozen=True)
class CombinedForceAndEquilibriumGapObjectiveResult:
    """Decomposed diagnostics for one objective evaluation."""

    total_cost: float
    equilibrium_gap_cost: float
    force_cost: float
    egi_scalars: npt.NDArray[np.float64]
    egi_baselines: npt.NDArray[np.float64]
    egi_window_weights: npt.NDArray[np.float64]
    force_scalar: float
    force_baseline: float
    force_weight: float


class CombinedForceAndEquilibriumGapObjective(IScalarObjectiveFunction):
    """Scalar objective combining normalised EGI and FRE contributions.

    ``phi = (1-lambda) sum(gamma_k * EGI_k / a_k)
    + lambda * FRE / b``.

    The EGI scalars are evaluated independently.  A combined EGI map remains
    useful for seeding basis centres, but is deliberately not used here: the
    RMS of a combined map is not equivalent to this scalar aggregation.
    """

    def __init__(
        self,
        *,
        force_weight: float = 0.1,
        egi_baseline_values: Sequence[float] | npt.NDArray[np.float64],
        force_baseline_value: float,
        egi_window_weights: Sequence[float] | npt.NDArray[np.float64],
    ) -> None:
        if not 0.0 <= force_weight <= 1.0:
            raise ValueError("force_weight must lie in [0, 1].")
        self.force_weight = float(force_weight)
        self.egi_baseline_values = _positive_vector(
            egi_baseline_values, "egi_baseline_values"
        )
        self.force_baseline_value = _positive_scalar(
            force_baseline_value, "force_baseline_value"
        )
        self.egi_window_weights = _normalised_weights(egi_window_weights)
        if self.egi_window_weights.shape != self.egi_baseline_values.shape:
            raise ValueError(
                "egi_window_weights and egi_baseline_values must have the "
                "same length."
            )
        self.last_result: CombinedForceAndEquilibriumGapObjectiveResult | None = None

    def evaluate(self, metric_results: list[MetricResult]) -> float:
        force_results, egi_results = _split_metric_results(metric_results)
        if len(force_results) != 1:
            raise ValueError("Objective requires exactly one FRE metric result.")
        if len(egi_results) != self.egi_baseline_values.size:
            raise ValueError(
                "Objective requires one EGI result per supplied baseline: "
                f"{len(egi_results)} vs {self.egi_baseline_values.size}."
            )

        egi_scalars = np.asarray(
            [_extract_egi_scalar(result) for result in egi_results],
            dtype=np.float64,
        )
        force_scalar = calculate_force_reconstruction_spatiotemporal_rms(
            force_results[0]
        )
        equilibrium_gap_cost = float(
            np.sum(
                self.egi_window_weights
                * egi_scalars
                / self.egi_baseline_values
            )
        )
        force_cost = float(force_scalar / self.force_baseline_value)
        total_cost = (
            (1.0 - self.force_weight) * equilibrium_gap_cost
            + self.force_weight * force_cost
        )
        self.last_result = CombinedForceAndEquilibriumGapObjectiveResult(
            total_cost=float(total_cost),
            equilibrium_gap_cost=equilibrium_gap_cost,
            force_cost=force_cost,
            egi_scalars=egi_scalars,
            egi_baselines=self.egi_baseline_values.copy(),
            egi_window_weights=self.egi_window_weights.copy(),
            force_scalar=float(force_scalar),
            force_baseline=self.force_baseline_value,
            force_weight=self.force_weight,
        )
        return float(total_cost)


def infer_egi_window_length_weights(
    metric_results: Sequence[MetricResult],
) -> npt.NDArray[np.float64]:
    """Return EGI window-length weights from EGI metadata.

    The paper uses mean window side length, rather than window area.  For a
    square 29/57 pair this gives weights 29/(29+57), 57/(29+57).
    """

    lengths = []
    for metric_result in metric_results:
        window_size = (metric_result.additional_fields or {}).get("window_size")
        if window_size is None:
            raise ValueError("EGI metric result does not contain window_size metadata.")
        dimensions = np.asarray(window_size, dtype=np.float64)
        if dimensions.shape != (2,) or np.any(dimensions <= 0.0):
            raise ValueError(f"Invalid EGI window_size metadata {dimensions}.")
        lengths.append(float(np.mean(dimensions)))
    return _normalised_weights(lengths)


def _split_metric_results(
    metric_results: Sequence[MetricResult],
) -> tuple[list[MetricResult], list[MetricResult]]:
    force_results: list[MetricResult] = []
    egi_results: list[MetricResult] = []
    for metric_result in metric_results:
        metadata = metric_result.additional_fields or {}
        if "weighted_spatiotemporal_rms" in metadata and "window_size" in metadata:
            egi_results.append(metric_result)
        elif "reconstructed_force" in metadata and "normalised_residual" in metadata:
            force_results.append(metric_result)
    return force_results, egi_results


def _extract_egi_scalar(metric_result: MetricResult) -> float:
    value = (metric_result.additional_fields or {}).get(
        "weighted_spatiotemporal_rms"
    )
    return _non_negative_scalar(value, "EGI weighted_spatiotemporal_rms")


def _positive_scalar(value: object, name: str) -> float:
    scalar = float(value)  # type: ignore[arg-type]
    if not np.isfinite(scalar) or scalar <= 0.0:
        raise ValueError(f"{name} must be finite and greater than zero.")
    return scalar


def _non_negative_scalar(value: object, name: str) -> float:
    scalar = float(value)  # type: ignore[arg-type]
    if not np.isfinite(scalar) or scalar < 0.0:
        raise ValueError(f"{name} must be finite and non-negative.")
    return scalar


def _positive_vector(value: Sequence[float] | npt.NDArray[np.float64], name: str) -> npt.NDArray[np.float64]:
    resolved = np.asarray(value, dtype=np.float64)
    if resolved.ndim != 1 or resolved.size == 0:
        raise ValueError(f"{name} must be a non-empty one-dimensional sequence.")
    if np.any(~np.isfinite(resolved)) or np.any(resolved <= 0.0):
        raise ValueError(f"{name} must contain finite values greater than zero.")
    return resolved


def _normalised_weights(value: Sequence[float] | npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    resolved = _positive_vector(value, "egi_window_weights")
    return resolved / float(np.sum(resolved))
