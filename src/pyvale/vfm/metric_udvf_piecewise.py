from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from pyvale.vfm.metrics import BaseMetric
from pyvale.vfm.project_definition import MetricSpec, TestData


@dataclass(slots=True)
class UDVFPiecewiseMetric(BaseMetric):
    options: dict[str, Any] = field(default_factory=dict)
    kind: str = "udvf_piecewise"

    def evaluate(self, stress, test_data: TestData, context=None):
        raise NotImplementedError(
            "Piecewise UDVF is scaffolded but not implemented yet."
        )


def build_udvf_piecewise_metric(metric_spec: MetricSpec) -> BaseMetric:
    return UDVFPiecewiseMetric(options=metric_spec.options)
