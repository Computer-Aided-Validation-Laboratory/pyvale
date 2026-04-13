from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from pyvale.vfm.metrics import BaseMetric
from pyvale.vfm.project_definition import MetricSpec, TestData


@dataclass(slots=True)
class UDVFUniformMetric(BaseMetric):
    options: dict[str, Any] = field(default_factory=dict)
    kind: str = "udvf_uniform"

    def evaluate(self, stress, test_data: TestData, context=None):
        raise NotImplementedError(
            "Uniform UDVF is scaffolded but not implemented yet."
        )


def build_udvf_uniform_metric(metric_spec: MetricSpec) -> BaseMetric:
    return UDVFUniformMetric(options=metric_spec.options)
