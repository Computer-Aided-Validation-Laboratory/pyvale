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

import pyvale.sensorsim as sens
import pyvale.valid as val

matplotlib.use("Agg")


def test_cdf_panels_identify_sources_and_preserve_point_selection() -> None:
    val_case = load_plate_case(transient=True)
    result = val.ValidationEngine([val.MetricMAVM()]).run(val_case)
    options = sens.PlotOptsGeneral(resolution=120, lw=2.5, font_ax_size=13)
    labels = val.ValPlotLabels(
        x_label="Thermocouple temperature [degC]",
        simulation="Nominal model",
        experiment="Synthetic measurements",
    )
    fig, axes = val.plot_val_cdfs_points(
        val_case,
        result,
        "temperature",
        point_idxs=(5, 0, 11),
        columns=2,
        time_idx=4,
        metric=val.MetricMAVM(),
        plot_opts=options,
        labels=labels,
    )
    assert axes.shape == (2, 2)
    assert not axes[1, 1].get_visible()
    assert fig.dpi == 120
    for point, ax in zip((5, 0, 11), axes.flat, strict=False):
        observations = val_case.sim["temperature"].select_aleatory(point, 0, 4)
        np.testing.assert_allclose(ax.lines[0].get_xdata()[1:], np.unique(observations))
        assert f"Point {point}" in ax.get_title()
        assert "d+=" in ax.get_title() and "d-=" in ax.get_title()
        assert ax.get_xlabel() == labels.x_label
        assert ax.title.get_fontsize() == options.font_head_size
        assert ax.xaxis.label.get_fontsize() == 13
        assert ax.lines[0].get_linewidth() == 2.5
        assert [text.get_text() for text in ax.get_legend().get_texts()][:2] == [
            labels.simulation,
            labels.experiment,
        ]
    plt.close(fig)


def test_metric_panels_select_direction_and_leave_selectors_unchanged() -> None:
    val_case = load_plate_case(transient=True, epistemic=True, offsets=True)
    result = val.ValidationEngine([val.MetricMAVM()]).run(val_case)
    selectors = {
        val.EValResAxis.SIM_TRANS: 1,
        val.EValResAxis.EXP_OFFSET: val.EEpisOffset.UPPER,
    }
    fig, axes = val.plot_val_metric_time_points(
        result,
        "temperature",
        val.EMetricType.MAVM,
        point_idxs=(0, 5),
        selectors=selectors,
        part=val.EValMetricPart.D_MINUS,
    )
    field = result.get("temperature", val.EMetricType.MAVM)
    for point, ax in zip((0, 5), axes.flat, strict=True):
        np.testing.assert_allclose(
            ax.lines[0].get_ydata(), field.d_minus[point, 0, :, 1, 2]
        )
        assert ax.get_ylabel() == "MAVM d− [degC]"
    assert val.EValResAxis.POINT not in selectors
    plt.close(fig)


def test_envelope_identifies_experiment_source() -> None:
    val_case = load_plate_case()
    fig, ax = val.plot_val_cdf_envelope(
        val_case,
        "temperature",
        source=val.EValSource.EXP,
    )
    assert all("Exp" in line.get_label() for line in ax.lines)
    plt.close(fig)


@pytest.mark.parametrize("points, columns", [((), 2), ((0,), 0)])
def test_point_panels_reject_empty_or_invalid_layout(
    points: tuple[int, ...],
    columns: int,
) -> None:
    val_case = load_plate_case()
    result = val.ValidationEngine([val.MetricAVM()]).run(val_case)
    with pytest.raises(ValueError, match="point"):
        val.plot_val_cdfs_points(
            val_case,
            result,
            "temperature",
            point_idxs=points,
            columns=columns,
        )


def test_envelope_grid_defaults_to_every_point_with_plot_options() -> None:
    val_case = load_plate_case(transient=True, epistemic=True, offsets=True)
    options = sens.PlotOptsGeneral(resolution=110, lw=2.0, font_head_size=11)
    labels = val.ValPlotLabels(
        title="Calibration envelope",
        x_label="Temperature [degC]",
        experiment="Measured temperatures",
    )
    fig, axes = val.plot_val_cdf_envelope_points(
        val_case,
        "temperature",
        source=val.EValSource.EXP,
        time_idx=4,
        plot_opts=options,
        labels=labels,
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
        assert ax.get_xlabel() == labels.x_label
        assert ax.lines[0].get_linewidth() == 2.0
        assert "Measured temperatures" in ax.lines[0].get_label()
        support, lower, upper = val.extract_ecdf_envelope(
            val_case.exp["temperature"],
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
    val_case = load_plate_case(epistemic=True)
    fig, axes = val.plot_val_cdf_envelope_points(
        val_case,
        "temperature",
        point_idxs=(11, 0, 5),
        epis_trans_idx=1,
        rows=rows,
        columns=columns,
    )
    assert axes.shape == shape
    for point, ax in zip((11, 0, 5), axes.flat, strict=False):
        expected = val_case.sim["temperature"].select_aleatory(
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
        val.plot_val_cdf_envelope_points(
            load_plate_case(),
            "temperature",
            point_idxs=points,
            rows=rows,
            columns=columns,
        )


def test_envelope_grid_supports_a_single_global_observable() -> None:
    observable_tag = "response"
    domain = val.ValDomain(
        {
            observable_tag: val.ValObservable(
                geometry=val.ValGeometry(val.EValGeometry.GLOBAL, None),
                operator=val.EValOperator.SAMPLE,
                components=("response",),
                phys_units="unit",
            )
        }
    )
    samples = {
        observable_tag: val.ValArray(
            np.array([0.0, 1.0, 2.0]), (val.EValAxis.ALEATORY,)
        )
    }
    fig, axes = val.plot_val_cdf_envelope_points(
        val.ValCase(domain, samples, samples),
        observable_tag,
    )
    assert axes.shape == (1, 1)
    assert "Point 0: response, steady" == axes[0, 0].get_title()
    plt.close(fig)
