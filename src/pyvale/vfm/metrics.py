from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import numpy.typing as npt

if TYPE_CHECKING:
    from pyvale.vfm.mechanical_properties import MechanicalProperties
    from pyvale.vfm.project_definition import MetricSpec, TestData
    from pyvale.vfm.spatial_parameterisation import ParameterState, ParameterisationDof
    from pyvale.vfm.project_definition import PhaseDefinition


@dataclass(slots=True)
class MetricResult:
    name: str
    value: float
    spatial_map: npt.NDArray | None = None
    spatial_temporal_map: npt.NDArray | None = None
    details: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class MetricContext:
    """Optional runtime information passed to a metric evaluation."""

    phase_definition: PhaseDefinition | None = None
    base_mechanical_properties: MechanicalProperties | None = None
    resolved_mechanical_properties: MechanicalProperties | None = None
    parameter_states: dict[str, ParameterState] | None = None
    active_dofs: list[ParameterisationDof] = field(default_factory=list)
    parameter_maps: dict[str, npt.NDArray] = field(default_factory=dict)


class BaseMetric(ABC):
    kind: str = "metric"

    def prepare(
        self,
        test_data: TestData,
        context: MetricContext | None = None,
    ) -> None:
        """Prepare reusable metric data before optimisation."""

    @abstractmethod
    def evaluate(
        self,
        stress: npt.NDArray,
        test_data: TestData,
        context: MetricContext | None = None,
    ) -> MetricResult:
        """Evaluate the metric for the current stress field."""


def build_metric(metric_spec: MetricSpec) -> BaseMetric:
    from pyvale.vfm.metric_equilibrium_gap_indicator import build_equilibrium_gap_metric
    from pyvale.vfm.metric_force_reconstruction_error import build_force_reconstruction_metric
    from pyvale.vfm.metric_sensitivity_based_vf import build_sensitivity_based_vf_metric
    from pyvale.vfm.metric_udvf_piecewise import build_udvf_piecewise_metric
    from pyvale.vfm.metric_udvf_slicewise import build_udvf_slicewise_metric
    from pyvale.vfm.metric_udvf_uniform import build_udvf_uniform_metric

    builders = {
        "sensitivity_based_vf": build_sensitivity_based_vf_metric,
        "sbvf": build_sensitivity_based_vf_metric,
        "equilibrium_gap_indicator": build_equilibrium_gap_metric,
        "egi": build_equilibrium_gap_metric,
        "force_reconstruction_error": build_force_reconstruction_metric,
        "fre": build_force_reconstruction_metric,
        "udvf_slicewise": build_udvf_slicewise_metric,
        "udvf_uniform": build_udvf_uniform_metric,
        "udvf_piecewise": build_udvf_piecewise_metric,
    }

    try:
        builder = builders[metric_spec.kind]
    except KeyError as error:
        raise ValueError(f"Unsupported metric kind '{metric_spec.kind}'.") from error

    return builder(metric_spec)


def evaluate_metrics(
    metrics_with_weights: list[tuple[BaseMetric, float]],
    stress: npt.NDArray,
    test_data: TestData,
    context: MetricContext | None = None,
) -> tuple[float, dict[str, float], list[MetricResult]]:
    total_cost = 0.0
    metric_values: dict[str, float] = {}
    results: list[MetricResult] = []

    for metric, weight in metrics_with_weights:
        result = metric.evaluate(stress, test_data, context)
        results.append(result)
        metric_values[result.name] = result.value
        total_cost += weight * result.value

    return total_cost, metric_values, results
