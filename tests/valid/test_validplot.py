# ==============================================================================
# pyvale: the python validation engine
# License: MIT
# Copyright (C) 2026 The Computer Aided Validation Team
# ==============================================================================

"""Plot controls and physical point selection for validation diagnostics."""

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pytest
from plate_fixture import load_plate_case

from pyvale import valid
from pyvale.valid import validplot

matplotlib.use("Agg")


def test_cdf_panels_identify_sources_and_preserve_point_selection() -> None:
    valid_case = load_plate_case(transient=True)
    result = valid.ValidEngine([valid.MetricMAVM()]).run(valid_case)
    options = validplot.ValidPlotOpts(
        resolution=120,
        lw=2.5,
        font_ax_size=13,
        x_label="Thermocouple temperature [degC]",
        sim_label="Nominal model",
        exp_label="Synthetic measurements",
    )
    fig, axes = validplot.plot_valid_cdfs_points(
        valid_case,
        result,
        "temperature",
        point_idxs=(5, 0, 11),
        columns=2,
        time_idx=4,
        metric=valid.MetricMAVM(),
        valid_plot_opts=options,
    )
    assert axes.shape == (2, 2)
    assert not axes[1, 1].get_visible()
    assert fig.dpi == 120
    for point, ax in zip((5, 0, 11), axes.flat, strict=False):
        observations = valid_case.sim["temperature"].select_aleatory(point, 0, 4)
        np.testing.assert_allclose(ax.lines[0].get_xdata()[1:], np.unique(observations))
        assert f"Point {point}" in ax.get_title()
        assert "d+=" in ax.get_title() and "d-=" in ax.get_title()
        assert ax.get_xlabel() == options.x_label
        assert ax.title.get_fontsize() == options.font_head_size
        assert ax.xaxis.label.get_fontsize() == 13
        assert ax.lines[0].get_linewidth() == 2.5
        assert [text.get_text() for text in ax.get_legend().get_texts()][:2] == [
            options.sim_label,
            options.exp_label,
        ]
    plt.close(fig)


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


def test_envelope_identifies_exp_source() -> None:
    valid_case = load_plate_case()
    fig, ax = validplot.plot_valid_cdf_envelope(
        valid_case,
        "temperature",
        source=valid.EValidSource.EXP,
    )
    assert all("Exp" in line.get_label() for line in ax.lines)
    plt.close(fig)


@pytest.mark.parametrize("points, columns", [((), 2), ((0,), 0)])
def test_point_panels_reject_empty_or_invalid_layout(
    points: tuple[int, ...],
    columns: int,
) -> None:
    valid_case = load_plate_case()
    result = valid.ValidEngine([valid.MetricAVM()]).run(valid_case)
    with pytest.raises(ValueError, match="point"):
        validplot.plot_valid_cdfs_points(
            valid_case,
            result,
            "temperature",
            point_idxs=points,
            columns=columns,
        )


def test_envelope_grid_defaults_to_every_point_with_plot_options() -> None:
    valid_case = load_plate_case(transient=True, epistemic=True, offsets=True)
    options = validplot.ValidPlotOpts(
        resolution=110,
        lw=2.0,
        font_head_size=11,
        title="Calibration envelope",
        x_label="Temperature [degC]",
        exp_label="Measured temperatures",
    )
    fig, axes = validplot.plot_valid_cdf_envelope_points(
        valid_case,
        "temperature",
        source=valid.EValidSource.EXP,
        time_idx=4,
        valid_plot_opts=options,
    )
    assert axes.shape == (3, 4)
    assert fig.dpi == 110
    assert fig.get_figwidth() > fig.get_figheight()
    width, height = options.single_fig_size_landscape
    np.testing.assert_allclose(fig.get_size_inches(), (width * 4, height * 3))
    for point, ax in enumerate(axes.flat):
        assert f"Point {point}: temperature, t=200 s" in ax.get_title()
        assert "Calibration envelope" in ax.get_title()
        assert ax.title.get_fontsize() == 11
        assert ax.get_xlabel() == options.x_label
        assert ax.lines[0].get_linewidth() == 2.0
        assert "Measured temperatures" in ax.lines[0].get_label()
        support, lower, upper = valid.extract_ecdf_envelope(
            valid_case.exp["temperature"],
            point_idx=point,
            time_idx=4,
        )
        np.testing.assert_array_equal(ax.lines[0].get_xdata()[1:], support)
        np.testing.assert_array_equal(ax.lines[0].get_ydata()[1:], lower)
        np.testing.assert_array_equal(ax.lines[1].get_ydata()[1:], upper)
    plt.close(fig)


@pytest.mark.parametrize(
    ("rows", "columns", "shape"),
    ((None, None, (2, 2)), (1, None, (1, 3)), (None, 2, (2, 2)), (2, 3, (2, 3))),
)
def test_envelope_grid_preserves_point_order_and_scenario_selection(
    rows: int | None,
    columns: int | None,
    shape: tuple[int, int],
) -> None:
    valid_case = load_plate_case(epistemic=True)
    fig, axes = validplot.plot_valid_cdf_envelope_points(
        valid_case,
        "temperature",
        point_idxs=(11, 0, 5),
        epis_trans_idx=1,
        rows=rows,
        columns=columns,
    )
    assert axes.shape == shape
    for point, ax in zip((11, 0, 5), axes.flat, strict=False):
        expected = valid_case.sim["temperature"].select_aleatory(
            point, 0, 0, epis_trans_idx=1
        )
        np.testing.assert_array_equal(ax.lines[0].get_xdata()[1:], np.unique(expected))
        assert f"Point {point}:" in ax.get_title()
    assert all(not ax.get_visible() for ax in axes.flat[3:])
    plt.close(fig)


@pytest.mark.parametrize(
    ("points", "rows", "columns"),
    (
        ((), None, None),
        ((-1,), None, None),
        ((12,), None, None),
        ((0, 1), 1, 1),
        ((0,), 0, None),
        ((0,), None, -1),
    ),
)
def test_envelope_grid_rejects_invalid_points_and_layout(
    points: tuple[int, ...],
    rows: int | None,
    columns: int | None,
) -> None:
    with pytest.raises(ValueError, match="[Pp]oint"):
        validplot.plot_valid_cdf_envelope_points(
            load_plate_case(),
            "temperature",
            point_idxs=points,
            rows=rows,
            columns=columns,
        )


def test_envelope_grid_supports_a_single_global_observable() -> None:
    observable_tag = "response"
    observables = {
        observable_tag: valid.ValidObservable(
            components=("response",),
            geometry=valid.EValidGeometry.GLOBAL,
            coords=None,
            phys_units="unit",
        )
    }
    samples = {
        observable_tag: valid.ValidArray(
            np.array([0.0, 1.0, 2.0]), (valid.EValidAxis.ALEATORY,)
        )
    }
    fig, axes = validplot.plot_valid_cdf_envelope_points(
        valid.ValidCase(observables, samples, samples),
        observable_tag,
    )
    assert axes.shape == (1, 1)
    assert "Point 0: response, steady" == axes[0, 0].get_title()
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


def test_source_defaults_and_confidence_shading_are_configurable() -> None:
    valid_case = load_plate_case()
    result = valid.ValidEngine([valid.MetricMAVM()]).run(valid_case)
    options = validplot.ValidPlotOpts(
        sim_color="purple",
        exp_color="green",
        sim_linestyle="--",
        exp_linestyle=":",
        bounds_color="red",
        bounds_alpha=0.37,
    )
    fig, ax = validplot.plot_valid_cdfs(
        valid_case,
        result,
        "temperature",
        metric=valid.MetricMAVM(),
        valid_plot_opts=options,
    )
    assert [line.get_label() for line in ax.lines] == [
        "Sim.",
        "Exp.",
        "Exp. conf. lower",
        "Exp. conf. upper",
    ]
    assert ax.lines[0].get_color() == "purple"
    assert ax.lines[0].get_linestyle() == "--"
    assert ax.lines[1].get_color() == "green"
    assert ax.lines[1].get_linestyle() == ":"
    assert len(ax.collections) == 1
    assert ax.collections[0].get_alpha() == 0.37
    np.testing.assert_allclose(
        ax.collections[0].get_facecolor()[0], (1.0, 0.0, 0.0, 0.37)
    )
    plt.close(fig)
    options.shade_bounds = False
    fig, ax = validplot.plot_valid_cdf_envelope(
        valid_case,
        "temperature",
        source=valid.EValidSource.EXP,
        valid_plot_opts=options,
    )
    assert not ax.collections
    assert all(line.get_color() == "green" for line in ax.lines)
    plt.close(fig)
