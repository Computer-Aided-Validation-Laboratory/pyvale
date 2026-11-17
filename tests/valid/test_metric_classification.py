# ==============================================================================
# pyvale: the python validation engine
# License: MIT
# Copyright (C) 2026 The Computer Aided Validation Team
# ==============================================================================

"""Metric classifications drive comparison requirements and physical plot units."""

from dataclasses import dataclass

import matplotlib.pyplot as plt
import numpy as np
import pytest

from pyvale import valid
from pyvale.valid import validplot


@pytest.mark.parametrize(
    ("metric", "scale", "comparison"),
    (
        (
            valid.MetricMAVM(),
            valid.EMetricScale.PHYSICAL,
            valid.EMetricComparison.DISTRIBUTIONAL,
        ),
        (
            valid.MetricAVM(),
            valid.EMetricScale.PHYSICAL,
            valid.EMetricComparison.DISTRIBUTIONAL,
        ),
        (
            valid.MetricKS(),
            valid.EMetricScale.DIMENSIONLESS_01,
            valid.EMetricComparison.DISTRIBUTIONAL,
        ),
        (
            valid.MetricCVM(),
            valid.EMetricScale.DIMENSIONLESS,
            valid.EMetricComparison.DISTRIBUTIONAL,
        ),
        (
            valid.MetricRMSE(),
            valid.EMetricScale.PHYSICAL,
            valid.EMetricComparison.PAIRED,
        ),
        (
            valid.MetricRelErr(),
            valid.EMetricScale.DIMENSIONLESS,
            valid.EMetricComparison.PAIRED,
        ),
    ),
)
def test_builtin_metric_classifications(
    metric: valid.IValidMetric,
    scale: valid.EMetricScale,
    comparison: valid.EMetricComparison,
) -> None:
    assert metric.get_scale() is scale
    assert metric.get_comparison() is comparison


def test_cvm_dimensionless_statistic_is_not_bounded_by_one() -> None:
    sim = np.arange(10.0)
    exp = sim + 20.0
    # For disjoint samples n=m=10, CvM = (2nm+1)/(6(n+m)) = 201/120.
    assert valid.MetricCVM().calc(sim, exp).value == pytest.approx(201 / 120)
    assert valid.MetricCVM().get_scale() is valid.EMetricScale.DIMENSIONLESS


@dataclass(slots=True)
class MetricSampleDifference(valid.IValidMetric):
    """Custom implementation whose metadata deliberately differs from its KS tag."""

    comparison: valid.EMetricComparison
    scale: valid.EMetricScale = valid.EMetricScale.PHYSICAL

    def get_kind(self) -> valid.EMetricType:
        return valid.EMetricType.KS

    def get_scale(self) -> valid.EMetricScale:
        return self.scale

    def get_comparison(self) -> valid.EMetricComparison:
        return self.comparison

    def get_outputs(self) -> tuple[valid.EMetricOutput, ...]:
        return (valid.EMetricOutput.STATISTIC,)

    def calc(self, sim: np.ndarray, exp: np.ndarray) -> valid.ScalarMetricResult:
        return valid.ScalarMetricResult(
            self.get_kind(), float(np.sum(sim) - np.sum(exp))
        )

    def extract_outputs(self, result: valid.MetricResult) -> tuple[float, ...]:
        assert isinstance(result, valid.ScalarMetricResult)
        return (result.value,)


def test_custom_paired_classification_uses_joint_missing_observation_mask() -> None:
    axes = (valid.EValidAxis.ALEATORY,)
    valid_case = valid.ValidCase(
        {
            "length": valid.ValidObservable(
                components=("length",),
                geometry=valid.EValidGeometry.GLOBAL,
                phys_units="m",
            )
        },
        {"length": valid.ValidArray(np.array([1.0, np.nan, 3.0]), axes)},
        {"length": valid.ValidArray(np.array([0.0, 20.0, np.nan]), axes)},
    )
    metric = MetricSampleDifference(valid.EMetricComparison.PAIRED)
    result = valid.ValidEngine([metric]).run(valid_case)
    field = result.metrics["length"][valid.EMetricType.KS]
    # Only the first pair is complete: 1-0=1, rather than (1+3)-(0+20).
    np.testing.assert_array_equal(
        field.outputs[valid.EMetricOutput.STATISTIC], [[[1.0]]]
    )


@pytest.mark.parametrize("scale", tuple(valid.EMetricScale))
def test_custom_calibration_classification_allows_unequal_counts_and_controls_units(
    scale: valid.EMetricScale,
) -> None:
    axes = (valid.EValidAxis.ALEATORY,)
    valid_case = valid.ValidCase(
        {
            "length": valid.ValidObservable(
                components=("length",),
                geometry=valid.EValidGeometry.GLOBAL,
                phys_units="m",
            )
        },
        {"length": valid.ValidArray(np.array([0.1, 0.3]), axes)},
        {"length": valid.ValidArray(np.array([0.0]), axes)},
    )
    metric = MetricSampleDifference(valid.EMetricComparison.CALIBRATION, scale)
    result = valid.ValidEngine([metric]).run(valid_case)
    field = result.metrics["length"][valid.EMetricType.KS]
    np.testing.assert_allclose(field.outputs[valid.EMetricOutput.STATISTIC], [[[0.4]]])
    fig, ax = validplot.plot_valid_metric_distribution(
        result, "length", valid.EMetricType.KS
    )
    try:
        units = "m" if scale is valid.EMetricScale.PHYSICAL else "1"
        assert ax.get_xlabel().endswith(f"[{units}]")
    finally:
        plt.close(fig)
    metric.comparison = valid.EMetricComparison.PAIRED
    with pytest.raises(ValueError, match="equal aleatory counts"):
        valid.ValidEngine([metric]).run(valid_case)
