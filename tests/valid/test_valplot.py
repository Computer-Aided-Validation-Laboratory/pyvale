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
        points=(5, 0, 11),
        columns=2,
        time=4,
        metric=val.MetricMAVM(),
        plot_opts=options,
        labels=labels,
    )
    assert axes.shape == (2, 2)
    assert not axes[1, 1].get_visible()
    assert fig.dpi == 120
    for point, ax in zip((5, 0, 11), axes.flat, strict=False):
        observations = val_case.sim.quantities["temperature"].select_aleatory(
            point, 0, 4
        )
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
        points=(0, 5),
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
            points=points,
            columns=columns,
        )
