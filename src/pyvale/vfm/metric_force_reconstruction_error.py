from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from pyvale.vfm.metrics import BaseMetric
from pyvale.vfm.project_definition import MetricSpec, TestData


@dataclass(slots=True)
class ForceReconstructionErrorMetric(BaseMetric):
    options: dict[str, Any] = field(default_factory=dict)
    kind: str = "force_reconstruction_error"

    def evaluate(self, stress, test_data: TestData, context=None):
        raise NotImplementedError(
            "Force reconstruction error is scaffolded but not implemented yet."
        )


def build_force_reconstruction_metric(metric_spec: MetricSpec) -> BaseMetric:
    return ForceReconstructionErrorMetric(options=metric_spec.options)
