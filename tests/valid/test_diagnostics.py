# ==============================================================================
# pyvale: the python validation engine
# License: MIT
# Copyright (C) 2026 The Computer Aided Validation Team
# ==============================================================================

"""Independent analytic checks for conditional distributions and MAVM support shifts."""

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pytest

import pyvale.valid as val

matplotlib.use("Agg")


def test_crossing_cdfs_distinguish_pointwise_from_support_sum_bounds() -> None:
    scenarios = [np.array([0.0, 10.0]), np.array([4.0, 5.0])]
    pointwise = val.calc_ecdf_summary(scenarios)
    selected = val.calc_ecdf_summary(scenarios, bounds=val.EValECDFBounds.SUPPORT_SUM)
    np.testing.assert_allclose(pointwise.probabilities, [0.5, 1.0])
    np.testing.assert_allclose(pointwise.lower, [0.0, 5.0])
    np.testing.assert_allclose(pointwise.upper, [4.0, 10.0])
    np.testing.assert_allclose(pointwise.nominal, [2.0, 7.5])
    assert pointwise.lower_idx is None
    assert pointwise.upper_idx is None
    assert selected.lower_idx == 1
    assert selected.upper_idx == 0
    np.testing.assert_allclose(selected.lower, [4.0, 5.0])
    np.testing.assert_allclose(selected.upper, [0.0, 10.0])


def test_quantile_summary_preserves_duplicate_mass_and_unequal_counts() -> None:
    samples = [np.array([0.0, 0.0, 2.0]), np.array([1.0, 3.0])]
    summary = val.calc_ecdf_summary(samples)
    np.testing.assert_allclose(summary.probabilities, [0.5, 2.0 / 3.0, 1.0])
    np.testing.assert_allclose(summary.lower, [0.0, 0.0, 2.0])
    np.testing.assert_allclose(summary.upper, [1.0, 3.0, 3.0])
    np.testing.assert_allclose(summary.nominal, [0.5, 1.5, 2.5])
    cdf = val.calc_ecdf(samples[0])
    np.testing.assert_allclose(
        val.evaluate_ecdf_quantiles(cdf, np.array([0.0, 2.0 / 3.0, 1.0])),
        [0.0, 0.0, 2.0],
    )


def test_endpoint_corrections_are_anchored_to_model_limits() -> None:
    correction = val.calc_mavm_correction(
        np.array([0.0]), np.array([10.0]), np.array([-2.0]), np.array([13.0])
    )
    assert correction.lower.d_minus == pytest.approx(2.0)
    assert correction.upper.d_plus == pytest.approx(3.0)
    assert correction.lower_exp_idx == 0
    assert correction.upper_exp_idx == 1
    # The largest endpoint distance is 13, but that is not the upper-bound shift.
    assert correction.comparisons[2].d_plus == pytest.approx(13.0)
    np.testing.assert_allclose(
        correction.lower.model_quantiles - correction.lower.d_minus, [-2.0]
    )
    np.testing.assert_allclose(
        correction.upper.model_quantiles + correction.upper.d_plus, [13.0]
    )


@pytest.mark.parametrize("view", tuple(val.EValMAVMDiagnostic))
def test_diagnostic_views_show_expected_sources_and_support_shifts(
    view: val.EValMAVMDiagnostic,
) -> None:
    simulation = val.calc_ecdf_summary([np.array([0.0]), np.array([10.0])])
    experiment = val.calc_ecdf_summary([np.array([-2.0]), np.array([13.0])])
    correction = val.calc_mavm_correction(
        np.array([0.0]), np.array([10.0]), np.array([-2.0]), np.array([13.0])
    )
    fig, ax = val.plot_val_mavm_diagnostic(
        simulation,
        experiment,
        correction=correction,
        view=view,
        labels=val.ValPlotLabels(x_label="Temperature [degC]"),
    )
    try:
        assert ax.get_xlabel() == "Temperature [degC]"
        legend = [text.get_text() for text in ax.get_legend().get_texts()]
        if view is val.EValMAVMDiagnostic.MODEL_CORRECTION:
            assert not any("Experiment" in text for text in legend)
        else:
            assert any("Experiment" in text for text in legend)
        if view in (
            val.EValMAVMDiagnostic.CORRECTION,
            val.EValMAVMDiagnostic.MODEL_CORRECTION,
        ):
            np.testing.assert_allclose(ax.lines[-2].get_xdata(), [-2.0, -2.0])
            np.testing.assert_allclose(ax.lines[-1].get_xdata(), [13.0, 13.0])
            assert "Outer MAVM support-shift band" in legend
    finally:
        plt.close(fig)


def test_family_plot_preserves_conditional_ecdf_probability_mass() -> None:
    quantity = val.ValArray(
        np.array([[0.0, 0.0, 2.0], [1.0, 3.0, 3.0]]),
        (val.EValAxis.EPISTEMIC_TRANS, val.EValAxis.ALEATORY),
    )
    fig, ax = val.plot_val_ecdf_family(quantity)
    np.testing.assert_allclose(ax.lines[0].get_ydata(), [0.0, 2.0 / 3.0, 1.0])
    np.testing.assert_allclose(ax.lines[1].get_ydata(), [0.0, 1.0 / 3.0, 1.0])
    plt.close(fig)


@pytest.mark.parametrize(
    "probabilities",
    [np.array([-0.1]), np.array([1.1]), np.array([np.nan]), np.ones((2, 2))],
)
def test_invalid_quantile_probabilities_fail(probabilities: np.ndarray) -> None:
    with pytest.raises(ValueError):
        val.evaluate_ecdf_quantiles(val.calc_ecdf(np.array([1.0, 2.0])), probabilities)


def test_missing_diagnostic_inputs_fail_before_plotting() -> None:
    with pytest.raises(ValueError):
        val.calc_ecdf_summary([])
    summary = val.calc_ecdf_summary([np.array([1.0, 2.0])])
    with pytest.raises(ValueError, match="correction"):
        val.plot_val_mavm_diagnostic(summary, view=val.EValMAVMDiagnostic.CORRECTION)
