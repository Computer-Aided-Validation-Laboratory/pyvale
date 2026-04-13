from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from pyvale.vfm.metrics import BaseMetric
from pyvale.vfm.project_definition import MetricSpec, TestData


@dataclass(slots=True)
class EquilibriumGapIndicatorMetric(BaseMetric):
    options: dict[str, Any] = field(default_factory=dict)
    kind: str = "equilibrium_gap_indicator"

    def evaluate(self, stress, test_data: TestData, context=None):
        raise NotImplementedError(
            "Equilibrium gap indicator is scaffolded but not implemented yet."
        )


def build_equilibrium_gap_metric(metric_spec: MetricSpec) -> BaseMetric:
    return EquilibriumGapIndicatorMetric(options=metric_spec.options)
