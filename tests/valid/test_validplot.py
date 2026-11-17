# ==============================================================================
# pyvale: the python validation engine
# License: MIT
# Copyright (C) 2026 The Computer Aided Validation Team
# ==============================================================================

"""Conditional ECDF views, physical selection and metric figure controls."""

import matplotlib.pyplot as plt
import numpy as np
import pytest
from plate_fixture import load_plate_case

from pyvale import valid
from pyvale.valid import validplot


def test_metric_panels_select_direction_and_leave_selectors_unchanged() -> None:
    valid_case = load_plate_case(transient=True, epistemic=True, offsets=True)
    result = valid.ValidEngine([valid.MetricMAVM()]).run(valid_case)
    selectors = {
        valid.EValidResAxis.SIM_TRANS: 1,
        valid.EValidResAxis.EXP_OFFSET: valid.EEpisOffset.UPPER,
    }
    fig, axes = validplot.plot_valid_metric_time_points(
        result,
        "temperature",
        valid.EMetricType.MAVM,
        point_idxs=(0, 5),
        selectors=selectors,
        output=valid.EMetricOutput.D_MINUS,
    )
    field = result.metrics["temperature"][valid.EMetricType.MAVM]
    for point, ax in zip((0, 5), axes.flat, strict=True):
        np.testing.assert_allclose(
            ax.lines[0].get_ydata(),
            field.outputs[valid.EMetricOutput.D_MINUS][point, 0, :, 1, 2],
        )
        assert ax.get_ylabel() == "MAVM d− [degC]"
    assert valid.EValidResAxis.POINT not in selectors
    plt.close(fig)


def test_directional_outputs_can_share_time_axes() -> None:
    case = load_plate_case(transient=True)
    result = valid.ValidEngine([valid.MetricMAVM()]).run(case)
    fig, ax = validplot.plot_valid_metric_time(
        result,
        "temperature",
        valid.EMetricType.MAVM,
        selectors={valid.EValidResAxis.POINT: 0},
        output=valid.EMetricOutput.D_PLUS,
    )
    validplot.plot_valid_metric_time(
        result,
        "temperature",
        valid.EMetricType.MAVM,
        selectors={valid.EValidResAxis.POINT: 0},
        output=valid.EMetricOutput.D_MINUS,
        ax=ax,
    )
    assert [text.get_text() for text in ax.get_legend().get_texts()] == [
        "MAVM d+",
        "MAVM d−",
    ]
    field = result.metrics["temperature"][valid.EMetricType.MAVM]
    for line, output in zip(
        ax.lines, (valid.EMetricOutput.D_PLUS, valid.EMetricOutput.D_MINUS), strict=True
    ):
        np.testing.assert_allclose(line.get_ydata(), field.outputs[output][0, 0])
    plt.close(fig)


def create_epistemic_case() -> valid.ValidCase:
    # Nuisance medians are [8,3,5]; their transferable median is 5.
    # Reversing reduction order would give median([2,6,7])=6 instead.
    shifts = np.array([[1.0, 9.0, 8.0], [2.0, 3.0, 7.0], [4.0, 6.0, 5.0]])
    samples = shifts[..., None] + np.array([0.0, 2.0])
    quantity = valid.ValidArray(
        samples,
        (
            valid.EValidAxis.EPISTEMIC_TRANS,
            valid.EValidAxis.EPISTEMIC_NUIS,
            valid.EValidAxis.ALEATORY,
        ),
    )
    observable = valid.ValidObservable(
        components=("length",), geometry=valid.EValidGeometry.GLOBAL, phys_units="m"
    )
    exp = valid.ValidArray(np.array([2.0, 5.0, 8.0]), (valid.EValidAxis.ALEATORY,))
    return valid.ValidCase(
        {"length": observable}, {"length": quantity}, {"length": exp}
    )


def test_quantile_reductions_preserve_axes_and_use_nuisance_first() -> None:
    case = create_epistemic_case()
    quantity = case.sim["length"]
    curves = valid.calc_ecdf_view(quantity)
    assert curves.quantiles.shape == (3, 3, 1, 2)
    np.testing.assert_array_equal(curves.probabilities, [0.5, 1.0])
    np.testing.assert_array_equal(curves.quantiles[..., 0, :], quantity.data)
    median_nuis = valid.calc_ecdf_view(
        quantity, view=valid.ValidECDFView(nuis_reduce=valid.EValidECDFReduce.MEDIAN)
    )
    np.testing.assert_array_equal(
        median_nuis.quantiles[:, 0, 0], [[8.0, 10.0], [3.0, 5.0], [5.0, 7.0]]
    )
    median_both = valid.calc_ecdf_view(
        quantity,
        view=valid.ValidECDFView(
            nuis_reduce=valid.EValidECDFReduce.MEDIAN,
            trans_reduce=valid.EValidECDFReduce.MEDIAN,
        ),
    )
    np.testing.assert_array_equal(median_both.quantiles, [[[[5.0, 7.0]]]])
    mean_both = valid.calc_ecdf_view(
        quantity,
        view=valid.ValidECDFView(
            nuis_reduce=valid.EValidECDFReduce.MEAN,
            trans_reduce=valid.EValidECDFReduce.MEAN,
        ),
    )
    np.testing.assert_allclose(mean_both.quantiles, [[[[5.0, 7.0]]]])


def test_selection_precedes_reduction_and_original_joint_overlays_are_preserved() -> (
    None
):
    quantity = create_epistemic_case().sim["length"]
    view = valid.ValidECDFView(
        trans_idxs=(2, 0),
        nuis_idxs=(1, 2),
        nuis_reduce=valid.EValidECDFReduce.MEAN,
        scenario_idxs=((0, 2), (2, 1)),
    )
    curves = valid.calc_ecdf_view(quantity, view=view)
    np.testing.assert_array_equal(curves.quantiles[:, 0, 0], [[5.5, 7.5], [8.5, 10.5]])
    assert curves.scenarios == (
        (0, 2, valid.EEpisOffset.NOMINAL),
        (2, 1, valid.EEpisOffset.NOMINAL),
    )
    np.testing.assert_array_equal(curves.scenario_quantiles, [[8.0, 10.0], [6.0, 8.0]])


def test_unequal_aleatory_counts_use_the_union_probability_grid() -> None:
    quantity = valid.ValidArray(
        np.array([[0.0, 10.0, np.nan], [2.0, 3.0, 20.0]]),
        (valid.EValidAxis.EPISTEMIC_TRANS, valid.EValidAxis.ALEATORY),
    )
    curves = valid.calc_ecdf_view(quantity)
    np.testing.assert_allclose(curves.probabilities, [1 / 3, 0.5, 2 / 3, 1.0])
    np.testing.assert_array_equal(
        curves.quantiles[:, 0, 0], [[0.0, 0.0, 10.0, 10.0], [2.0, 3.0, 3.0, 20.0]]
    )


def test_crossing_scenarios_form_pointwise_bounds_not_two_selected_scenarios() -> None:
    quantity = valid.ValidArray(
        np.array([[0.0, 10.0], [4.0, 6.0]]),
        (valid.EValidAxis.EPISTEMIC_TRANS, valid.EValidAxis.ALEATORY),
    )
    obs = valid.ValidObservable(
        components=("length",), geometry=valid.EValidGeometry.GLOBAL
    )
    case = valid.ValidCase({"length": obs}, {"length": quantity}, {"length": quantity})
    ecdf_plot: validplot.ValidECDFPlot = validplot.plot_sim_ecdfs(case, "length")
    (fig, axes) = ecdf_plot.figure, ecdf_plot.axes
    ax = axes[0, 0]
    np.testing.assert_array_equal(ax.lines[0].get_xdata()[1:], [0.0, 6.0])
    np.testing.assert_array_equal(ax.lines[1].get_xdata()[1:], [4.0, 10.0])
    assert len(ax.collections) == 1
    assert "confidence" not in " ".join(line.get_label() for line in ax.lines)
    plt.close(fig)


def test_offsets_are_explicit_and_retained_as_coherent_scenarios() -> None:
    quantity = valid.ValidArray(
        np.array([1.0, 3.0]),
        (valid.EValidAxis.ALEATORY,),
        valid.EpistemicOffsets(0.5, 2.0),
    )
    nominal = valid.calc_ecdf_view(quantity)
    assert nominal.quantiles.shape == (1, 1, 1, 2)
    np.testing.assert_array_equal(nominal.quantiles[0, 0, 0], [1.0, 3.0])
    all_offsets = valid.calc_ecdf_view(
        quantity, offset=None, view=valid.ValidECDFView(scenario_idxs=None)
    )
    assert all_offsets.quantiles.shape == (1, 1, 3, 2)
    np.testing.assert_array_equal(
        all_offsets.quantiles[0, 0], [[1.0, 3.0], [0.5, 2.5], [3.0, 5.0]]
    )
    assert tuple(scenario[2] for scenario in all_offsets.scenarios) == tuple(
        valid.EEpisOffset
    )


@pytest.mark.parametrize(
    "centre, expected",
    [
        (valid.EValidECDFCentre.MEAN, [5.0, 7.0]),
        (valid.EValidECDFCentre.MEDIAN, [5.0, 7.0]),
    ],
)
def test_centre_curves_and_selected_scenarios_can_be_combined(
    centre: valid.EValidECDFCentre, expected: list[float]
) -> None:
    view = valid.ValidECDFView(centre=centre, scenario_idxs=((0, 0), (2, 2)))
    ecdf_plot: validplot.ValidECDFPlot = validplot.plot_sim_ecdfs(
        create_epistemic_case(), "length", view=view
    )
    (fig, axes) = ecdf_plot.figure, ecdf_plot.axes
    ax = axes[0, 0]
    assert len(ax.lines) == 5  # Two bounds, one centre, two original curves.
    np.testing.assert_allclose(ax.lines[2].get_xdata()[1:], expected)
    assert ax.lines[2].get_label() == f"Sim. {centre.name.lower()} quantile"
    np.testing.assert_array_equal(ax.lines[3].get_xdata()[1:], [1.0, 3.0])
    np.testing.assert_array_equal(ax.lines[4].get_xdata()[1:], [5.0, 7.0])
    plt.close(fig)


def test_independent_sim_exp_views_and_single_scenario_have_no_duplicate_shading() -> (
    None
):
    case = create_epistemic_case()
    ecdf_plot: validplot.ValidECDFPlot = validplot.plot_sim_exp_cdfs(
        case,
        "length",
        sim_view=valid.ValidECDFView(nuis_reduce=valid.EValidECDFReduce.MEDIAN),
    )
    (fig, axes) = ecdf_plot.figure, ecdf_plot.axes
    ax = axes[0, 0]
    assert len(ax.lines) == 3
    np.testing.assert_array_equal(ax.lines[0].get_xdata()[1:], [3.0, 5.0])
    np.testing.assert_array_equal(ax.lines[1].get_xdata()[1:], [8.0, 10.0])
    assert ax.lines[2].get_label() == "Exp."
    assert len(ax.collections) == 1
    assert "d+=" not in ax.get_title()
    plt.close(fig)
    ecdf_plot: validplot.ValidECDFPlot = validplot.plot_exp_ecdfs(
        case, "length", view=valid.ValidECDFView(scenario_idxs=None)
    )
    (fig, axes) = ecdf_plot.figure, ecdf_plot.axes
    assert len(axes[0, 0].lines) == 1
    assert not axes[0, 0].collections
    plt.close(fig)


def test_many_scenarios_default_to_two_bounds_and_honour_style_controls() -> None:
    case = create_epistemic_case()
    case.sim["length"] = valid.ValidArray(
        np.arange(1000.0)[:, None] + np.array([0.0, 1.0]),
        (valid.EValidAxis.EPISTEMIC_TRANS, valid.EValidAxis.ALEATORY),
    )
    options = validplot.ValidPlotOpts(
        sim_color="purple", bounds_color="red", bounds_alpha=0.37, lower_linestyle="-."
    )
    ecdf_plot: validplot.ValidECDFPlot = validplot.plot_sim_ecdfs(
        case, "length", valid_plot_opts=options
    )
    (fig, axes) = ecdf_plot.figure, ecdf_plot.axes
    ax = axes[0, 0]
    assert len(ax.lines) == 2
    assert ax.lines[0].get_color() == "purple"
    assert ax.lines[0].get_linestyle() == "-."
    np.testing.assert_allclose(
        ax.collections[0].get_facecolor()[0], [1.0, 0.0, 0.0, 0.37]
    )
    plt.close(fig)


def test_panels_preserve_point_order_and_general_figure_options() -> None:
    case = load_plate_case(transient=True, epistemic=True)
    options = validplot.ValidPlotOpts(
        resolution=120, lw=2.5, font_ax_size=13, x_label="Temperature [degC]"
    )
    ecdf_plot: validplot.ValidECDFPlot = validplot.plot_sim_exp_cdfs(
        case,
        "temperature",
        point_idxs=(5, 0, 11),
        columns=2,
        time_idx=4,
        valid_plot_opts=options,
    )
    (fig, axes) = ecdf_plot.figure, ecdf_plot.axes
    assert axes.shape == (2, 2)
    assert not axes[1, 1].get_visible()
    assert fig.dpi == 120
    for point, ax in zip((5, 0, 11), axes.flat, strict=False):
        assert f"Point {point}" in ax.get_title()
        assert ax.get_xlabel() == options.x_label
        assert ax.xaxis.label.get_fontsize() == 13
        assert ax.lines[0].get_linewidth() == 2.5
        curves = valid.calc_ecdf_view(
            case.sim["temperature"], point_idx=point, time_idx=4
        )
        np.testing.assert_allclose(
            ax.lines[0].get_xdata()[1:], np.min(curves.quantiles, axis=(0, 1, 2))
        )
    plt.close(fig)
    ecdf_plot: validplot.ValidECDFPlot = validplot.plot_exp_ecdfs(
        case, "temperature", point_idxs=None, time_idx=4
    )
    (fig, axes) = ecdf_plot.figure, ecdf_plot.axes
    assert axes.size == 12
    plt.close(fig)


@pytest.mark.parametrize(
    "points, exception",
    [
        ((), ValueError),
        ((-1,), IndexError),
        ((12,), IndexError),
        ((True,), TypeError),
        ((0.5,), TypeError),
    ],
)
def test_invalid_points_do_not_create_figures(
    points: tuple[int, ...], exception: type[Exception]
) -> None:
    figures = plt.get_fignums()
    with pytest.raises(exception):
        validplot.plot_sim_ecdfs(load_plate_case(), "temperature", point_idxs=points)
    assert plt.get_fignums() == figures


@pytest.mark.parametrize(
    "view, exception",
    [
        (valid.ValidECDFView(trans_idxs=()), ValueError),
        (valid.ValidECDFView(trans_idxs=(3,)), IndexError),
        (valid.ValidECDFView(trans_idxs=(0, 0)), ValueError),
        (valid.ValidECDFView(trans_idxs=(True,)), TypeError),
        (valid.ValidECDFView(scenario_idxs=((0, 3),)), IndexError),
        (valid.ValidECDFView(scenario_idxs=((0, 0), (0, 0))), ValueError),
        (valid.ValidECDFView(bounds=False), ValueError),
    ],
)
def test_invalid_views_fail_before_plotting(
    view: valid.ValidECDFView, exception: type[Exception]
) -> None:
    figures = plt.get_fignums()
    with pytest.raises(exception):
        validplot.plot_sim_ecdfs(create_epistemic_case(), "length", view=view)
    assert plt.get_fignums() == figures


def test_existing_axes_support_composition_but_reject_multiple_points() -> None:
    fig, ax = plt.subplots()
    ecdf_plot: validplot.ValidECDFPlot = validplot.plot_sim_ecdfs(
        create_epistemic_case(), "length", ax=ax
    )
    (returned, axes) = ecdf_plot.figure, ecdf_plot.axes
    assert returned is fig and axes[0, 0] is ax
    with pytest.raises(ValueError, match="one point"):
        validplot.plot_sim_ecdfs(
            load_plate_case(), "temperature", point_idxs=(0, 1), ax=ax
        )
    plt.close(fig)
