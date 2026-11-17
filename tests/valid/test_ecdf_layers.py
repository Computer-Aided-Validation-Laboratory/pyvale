# ==============================================================================
# pyvale: the python validation engine
# License: MIT
# Copyright (C) 2026 The Computer Aided Validation Team
# ==============================================================================

"""Analytic geometry and coordinate checks for composable ECDF layers."""

import matplotlib.pyplot as plt
import numpy as np
import pytest

from pyvale import valid
from pyvale.valid import validplot


def create_case(sim: np.ndarray, exp: np.ndarray) -> valid.ValidCase:
    sim_axes = (valid.EValidAxis.ALEATORY,)
    if sim.ndim == 2:
        sim_axes = (valid.EValidAxis.EPISTEMIC_TRANS, valid.EValidAxis.ALEATORY)
    exp_axes = (valid.EValidAxis.ALEATORY,)
    if exp.ndim == 2:
        exp_axes = (valid.EValidAxis.EPISTEMIC_TRANS, valid.EValidAxis.ALEATORY)
    observable = valid.ValidObservable(
        components=("length",), geometry=valid.EValidGeometry.GLOBAL, phys_units="m"
    )
    return valid.ValidCase(
        {"length": observable},
        {"length": valid.ValidArray(sim, sim_axes)},
        {"length": valid.ValidArray(exp, exp_axes)},
    )


@pytest.mark.parametrize("probability, expected", [(0.5, 5.0), (0.75, 7.0), (1.0, 7.0)])
def test_reference_quantile_normalisation_is_a_constant_secondary_axis(
    probability: float, expected: float
) -> None:
    case = create_case(np.array([[0.0, 2.0], [10.0, 12.0]]), np.array([1.0, 3.0]))
    plot = validplot.plot_sim_exp_cdfs(case, "length")
    original_quantiles = plot.axes[0, 0].lines[0].get_xdata().copy()
    validplot.add_normalised_ecdf_axis(
        plot, validplot.ValidECDFReference(probability=probability)
    )
    plot.figure.canvas.draw()
    assert plot.normalisation == (expected,)
    np.testing.assert_array_equal(
        plot.axes[0, 0].lines[0].get_xdata(), original_quantiles
    )
    secondary = plot.axes[0, 0].child_axes[0]
    np.testing.assert_allclose(
        secondary.get_xlim(), np.asarray(plot.axes[0, 0].get_xlim()) / expected
    )
    with pytest.raises(ValueError, match="already"):
        validplot.add_normalised_ecdf_axis(plot, 2.0)
    plt.close(plot.figure)


def test_constant_and_exp_reference_scalings_are_explicit() -> None:
    case = create_case(np.array([2.0, 4.0]), np.array([10.0, 20.0]))
    plot = validplot.plot_sim_exp_cdfs(case, "length")
    validplot.add_normalised_ecdf_axis(
        plot, validplot.ValidECDFReference(source=valid.EValidSource.EXP)
    )
    assert plot.normalisation == (10.0,)
    plt.close(plot.figure)
    plot = validplot.plot_sim_ecdfs(case, "length")
    validplot.add_normalised_ecdf_axis(plot, 4.0)
    assert plot.normalisation == (4.0,)
    plt.close(plot.figure)


@pytest.mark.parametrize(
    "reference",
    [
        0.0,
        np.inf,
        np.nan,
        validplot.ValidECDFReference(probability=0.0),
        validplot.ValidECDFReference(probability=1.1),
    ],
)
def test_invalid_normalisation_does_not_add_partial_axes(
    reference: float | validplot.ValidECDFReference,
) -> None:
    plot = validplot.plot_sim_exp_cdfs(
        create_case(np.array([0.0, 2.0]), np.array([1.0, 3.0])), "length"
    )
    with pytest.raises(ValueError):
        validplot.add_normalised_ecdf_axis(plot, reference)
    assert plot.normalisation is None
    assert not plot.axes[0, 0].child_axes
    plt.close(plot.figure)


def test_zero_reference_quantile_is_rejected() -> None:
    plot = validplot.plot_sim_exp_cdfs(
        create_case(np.array([0.0, 2.0]), np.array([1.0, 3.0])), "length"
    )
    with pytest.raises(ValueError, match="nonzero"):
        validplot.add_normalised_ecdf_axis(plot, validplot.ValidECDFReference())
    plt.close(plot.figure)


@pytest.mark.parametrize(
    "output, exp, expected",
    [
        (valid.EMetricOutput.D_PLUS, np.array([11.0, 13.0]), np.array([11.0, 13.0])),
        (valid.EMetricOutput.D_MINUS, np.array([-1.0, 1.0]), np.array([-1.0, 1.0])),
    ],
)
def test_mavm_support_uses_outer_shifted_ecdfs_not_global_largest_distance(
    output: valid.EMetricOutput, exp: np.ndarray, expected: np.ndarray
) -> None:
    case = create_case(np.array([[0.0, 2.0], [10.0, 12.0]]), exp)
    plot = validplot.plot_sim_exp_cdfs(case, "length")
    validplot.add_metric_overlay(plot, valid.MetricMAVM(alpha=0.5), output=output)
    ax = plot.axes[0, 0]
    shifted_line = (
        ax.lines[-1] if output is valid.EMetricOutput.D_PLUS else ax.lines[-2]
    )
    np.testing.assert_allclose(
        shifted_line.get_xdata()[1:], expected, rtol=0, atol=1e-12
    )
    # The largest distance is 11, but adding it to the outer sim curve would
    # wrongly produce [21,23] (or subtracting it would produce [-11,-9]).
    assert "outer support shift" in ax.collections[-1].get_label()
    plt.close(plot.figure)


@pytest.mark.parametrize(
    "metric, output, expected_area",
    [
        (valid.MetricAVM(), valid.EMetricOutput.DISTANCE, 1.0),
        (valid.MetricMAVM(alpha=0.5), valid.EMetricOutput.D_PLUS, 0.5),
        (valid.MetricMAVM(alpha=0.5), valid.EMetricOutput.D_MINUS, 0.5),
    ],
)
def test_integration_polygon_area_equals_the_metric(
    metric: valid.IValidMetric, output: valid.EMetricOutput, expected_area: float
) -> None:
    sim = (
        np.array([0.0, 2.0])
        if isinstance(metric, valid.MetricAVM)
        else np.array([0.0, 4.0])
    )
    plot = validplot.plot_sim_exp_cdfs(create_case(sim, np.array([1.0, 3.0])), "length")
    validplot.add_metric_overlay(
        plot,
        metric,
        output=output,
        display=validplot.EValidMetricDisplay.INTEGRATION_AREA,
    )
    area = 0.0
    for path in plot.axes[0, 0].collections[-1].get_paths():
        coords = path.vertices
        area += (
            abs(
                np.dot(coords[:, 0], np.roll(coords[:, 1], -1))
                - np.dot(coords[:, 1], np.roll(coords[:, 0], -1))
            )
            / 2
        )
    assert area == pytest.approx(expected_area, rel=0, abs=1e-12)
    plt.close(plot.figure)


def test_multi_scenario_integration_requires_an_explicit_original_comparison() -> None:
    case = create_case(np.array([[0.0, 2.0], [10.0, 12.0]]), np.array([1.0, 3.0]))
    plot = validplot.plot_sim_exp_cdfs(case, "length")
    with pytest.raises(ValueError, match="explicit original"):
        validplot.add_metric_overlay(
            plot,
            valid.MetricAVM(),
            display=validplot.EValidMetricDisplay.INTEGRATION_AREA,
        )
    validplot.add_metric_overlay(
        plot,
        valid.MetricAVM(),
        display=validplot.EValidMetricDisplay.INTEGRATION_AREA,
        comparison=validplot.ValidECDFComparison(),
    )
    plt.close(plot.figure)


def test_ks_segments_show_exact_single_and_bound_cdf_differences() -> None:
    plot = validplot.plot_sim_exp_cdfs(
        create_case(np.array([0.0, 2.0]), np.array([1.0, 3.0])), "length"
    )
    validplot.add_metric_overlay(plot, valid.MetricKS())
    line = plot.axes[0, 0].lines[-1]
    assert np.ptp(line.get_xdata()) == 0
    assert np.ptp(line.get_ydata()) == pytest.approx(0.5)
    plt.close(plot.figure)
    plot = validplot.plot_sim_exp_cdfs(
        create_case(np.array([[0.0, 2.0], [10.0, 12.0]]), np.array([1.0, 3.0])),
        "length",
    )
    validplot.add_metric_overlay(plot, valid.MetricKS())
    labels = [line.get_label() for line in plot.axes[0, 0].lines[-2:]]
    assert "lower-support bounds" in labels[0]
    assert "upper-support bounds" in labels[1]
    plt.close(plot.figure)


def test_nongeometric_metrics_compose_as_annotations_without_reinterpreting_area() -> (
    None
):
    plot = validplot.plot_sim_exp_cdfs(
        create_case(np.array([0.0, 2.0]), np.array([1.0, 3.0])), "length"
    )
    for metric in (valid.MetricCVM(), valid.MetricRMSE(), valid.MetricRelErr()):
        validplot.add_metric_overlay(plot, metric)
    assert len(plot.axes[0, 0].texts) == 3
    assert not plot.axes[0, 0].collections
    with pytest.raises(ValueError, match="AVM or MAVM"):
        validplot.add_metric_overlay(
            plot,
            valid.MetricRMSE(),
            display=validplot.EValidMetricDisplay.SUPPORT_SHIFT,
        )
    plt.close(plot.figure)


def test_u_pooling_has_its_own_probability_coordinate_plot() -> None:
    cdf = valid.ValidECDF(np.array([0.0, 2.0]), np.array([0.5, 1.0]))
    fig, ax = validplot.plot_u_pooling_cdf([cdf], np.array([1.0]))
    label = ax.lines[0].get_label()
    assert isinstance(label, str)
    assert "0.25" in label
    assert ax.get_xlim() == (0.0, 1.0)
    assert "[1]" in ax.get_xlabel()
    plt.close(fig)


@pytest.mark.parametrize(
    "output, expected",
    [
        (valid.EMetricOutput.D_PLUS, [55.0, 115.0]),
        (valid.EMetricOutput.D_MINUS, [-15.0, 45.0]),
    ],
)
def test_shift_bounds_can_switch_controlling_scenario_with_both_epistemic_axes(
    output: valid.EMetricOutput,
    expected: list[float],
) -> None:
    # Every sim/exp source has both axes. Repeated transferable settings make
    # the controlling nuisance switch analytically visible: Q=[0,100] versus
    # Q=[40,60]. Comparisons with [10,90] and [30,70] give max d+=d-=15
    # for each sim curve, so different nuisance cases control each endpoint.
    axes = (
        valid.EValidAxis.EPISTEMIC_TRANS,
        valid.EValidAxis.EPISTEMIC_NUIS,
        valid.EValidAxis.ALEATORY,
    )
    sim = valid.ValidArray(
        np.tile(np.array([[[0.0, 100.0], [40.0, 60.0]]]), (2, 1, 1)), axes
    )
    exp = valid.ValidArray(
        np.tile(np.array([[[10.0, 90.0], [30.0, 70.0]]]), (2, 1, 1)), axes
    )
    observable = valid.ValidObservable(
        components=("length",), geometry=valid.EValidGeometry.GLOBAL
    )
    case = valid.ValidCase({"length": observable}, {"length": sim}, {"length": exp})
    plot = validplot.plot_sim_exp_cdfs(case, "length")
    validplot.add_metric_overlay(plot, valid.MetricMAVM(alpha=0.5), output=output)
    line = (
        plot.axes[0, 0].lines[-1]
        if output is valid.EMetricOutput.D_PLUS
        else plot.axes[0, 0].lines[-2]
    )
    np.testing.assert_allclose(line.get_xdata()[1:], expected, rtol=0, atol=1e-12)
    plt.close(plot.figure)


def test_avm_shift_is_a_symmetric_magnitude_about_each_original_curve() -> None:
    plot = validplot.plot_sim_exp_cdfs(
        create_case(np.array([[0.0, 2.0], [10.0, 12.0]]), np.array([11.0, 13.0])),
        "length",
    )
    validplot.add_metric_overlay(plot, valid.MetricAVM())
    np.testing.assert_allclose(plot.axes[0, 0].lines[-2].get_xdata()[1:], [-11.0, -9.0])
    np.testing.assert_allclose(plot.axes[0, 0].lines[-1].get_xdata()[1:], [11.0, 13.0])
    assert "symmetric" in plot.axes[0, 0].collections[-1].get_label()
    plt.close(plot.figure)


def test_precomputed_support_profiles_compose_without_metric_recalculation() -> None:
    case = create_case(np.array([[0.0, 2.0], [10.0, 12.0]]), np.array([11.0, 13.0]))
    result = valid.ValidEngine([valid.MetricMAVM(alpha=0.5)]).run(case)
    supports = valid.find_metric_epis_supports(
        case, result, "length", valid.EMetricType.MAVM
    )
    plot = validplot.plot_sim_exp_cdfs(case, "length")
    validplot.add_support_overlay(plot, supports, output=valid.EMetricOutput.D_PLUS)
    np.testing.assert_allclose(plot.axes[0, 0].lines[-1].get_xdata()[1:], [11.0, 13.0])
    plt.close(plot.figure)
