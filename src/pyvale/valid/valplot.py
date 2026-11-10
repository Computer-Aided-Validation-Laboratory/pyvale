# ==============================================================================
# pyvale: the python validation engine
# License: MIT
# Copyright (C) 2026 The Computer Aided Validation Team
# ==============================================================================

"""Diagnostic plots for validation samples and labelled metric fields."""

from dataclasses import dataclass

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import MaxNLocator, ScalarFormatter

from pyvale.sensorsim.visualopts import PlotOptsGeneral
from pyvale.valid.metrics import calc_mavm_1d
from pyvale.valid.metricstrategy import EMetricType, MetricMAVM
from pyvale.valid.valengine import (
    EValResAxis,
    ValMetricField,
    ValResult,
)
from pyvale.valid.valtools import calc_ecdf, extract_ecdf_envelope
from pyvale.valid.valtypes import EEpisOffset, EValSource, ValCase


@dataclass(slots=True)
class ValPlotLabels:
    """Override automatic observable/unit labels and source legend entries."""

    title: str | None = None
    x_label: str | None = None
    y_label: str | None = None
    colorbar_label: str | None = None
    simulation: str = "Simulation ECDF"
    experiment: str = "Experiment ECDF"
    exp_conf_lower: str = "Experiment lower confidence bound"
    exp_conf_upper: str = "Experiment upper confidence bound"


def _apply_plot_options(
    fig: plt.Figure,
    ax: plt.Axes,
    plot_opts: PlotOptsGeneral | None,
    labels: ValPlotLabels | None,
) -> None:
    if labels is not None:
        if labels.title is not None:
            ax.set_title(labels.title)
        if labels.x_label is not None:
            ax.set_xlabel(labels.x_label)
        if labels.y_label is not None:
            ax.set_ylabel(labels.y_label)

    options = PlotOptsGeneral() if plot_opts is None else plot_opts
    fig.set_dpi(options.resolution)
    ax.tick_params(labelsize=options.font_tick_size)
    ax.xaxis.set_major_locator(MaxNLocator(nbins=4))
    ax.xaxis.set_major_formatter(ScalarFormatter(useOffset=True))
    ax.xaxis.get_offset_text().set_fontsize(options.font_tick_size)
    for text in (ax.title, ax.xaxis.label, ax.yaxis.label):
        text.set_fontname(options.font_name)
        text.set_fontweight(options.font_def_weight)
    ax.title.set_fontsize(options.font_head_size)
    ax.xaxis.label.set_fontsize(options.font_ax_size)
    ax.yaxis.label.set_fontsize(options.font_ax_size)
    for line in ax.lines:
        line.set_linewidth(options.lw)
        line.set_markersize(options.ms)
    legend = ax.get_legend()
    if legend is not None:
        for text in legend.get_texts():
            text.set_fontsize(options.font_leg_size)
            text.set_fontname(options.font_name)


def _create_axes(
    ax: plt.Axes | None, plot_opts: PlotOptsGeneral | None
) -> tuple[plt.Figure, plt.Axes]:
    if ax is not None:
        return ax.figure, ax
    options = PlotOptsGeneral() if plot_opts is None else plot_opts
    return plt.subplots(figsize=options.single_fig_size_landscape, layout="constrained")


def _metric_label(metric: EMetricType, direction: str) -> str:
    if metric is EMetricType.MAVM:
        return {"metric": "MAVM d+ + d−", "d_plus": "MAVM d+", "d_minus": "MAVM d−"}[
            direction
        ]
    return metric.name


def _metric_units(result: ValResult, observable: str, metric: EMetricType) -> str:
    if metric in (EMetricType.KS, EMetricType.CVM, EMetricType.RELATIVE_ERROR):
        return "1"
    return result.domain.observables[observable].units


def plot_val_cdfs(
    val_case: ValCase,
    result: ValResult,
    observable: str,
    *,
    point: int = 0,
    component: int = 0,
    time: int = 0,
    sim_trans: int = 0,
    sim_nuis: int = 0,
    exp_trans: int = 0,
    exp_nuis: int = 0,
    sim_offset: EEpisOffset = EEpisOffset.NOMINAL,
    exp_offset: EEpisOffset = EEpisOffset.NOMINAL,
    metric: MetricMAVM | None = None,
    plot_opts: PlotOptsGeneral | None = None,
    labels: ValPlotLabels | None = None,
    ax: plt.Axes | None = None,
) -> tuple[plt.Figure, plt.Axes]:
    """Show the exact conditional samples compared at one physical cell."""
    if observable not in result.fields:
        raise KeyError(observable)
    sim = val_case.sim.quantities[observable].select_aleatory(
        point, component, time, sim_trans, sim_nuis, sim_offset
    )
    exp = val_case.exp.quantities[observable].select_aleatory(
        point, component, time, exp_trans, exp_nuis, exp_offset
    )
    sim = sim[np.isfinite(sim)]
    exp = exp[np.isfinite(exp)]
    if len(sim) == 0 or len(exp) == 0:
        raise ValueError("Selected cell has no finite observations.")
    raw = None
    support_min = min(np.min(sim), np.min(exp))
    support_max = max(np.max(sim), np.max(exp))
    if metric is not None:
        raw = calc_mavm_1d(sim, exp, metric.alpha, metric.duplicates, metric.tol)
        support_min = min(support_min, np.min(raw.exp_conf_lower))
        support_max = max(support_max, np.max(raw.exp_conf_upper))
    margin = max((support_max - support_min) * 0.02, 1.0e-12)
    left = support_min - margin
    fig, ax = _create_axes(ax, plot_opts)
    source_labels = ValPlotLabels() if labels is None else labels
    options = PlotOptsGeneral() if plot_opts is None else plot_opts
    for samples, label, color in (
        (sim, source_labels.simulation, options.colors[0]),
        (exp, source_labels.experiment, options.colors[1]),
    ):
        cdf = calc_ecdf(samples)
        ax.step(
            np.r_[left, cdf.quantiles],
            np.r_[0.0, cdf.probabilities],
            where="post",
            label=label,
            color=color,
        )
    if raw is not None:
        ax.step(
            np.r_[left, raw.exp_conf_lower],
            np.r_[0.0, raw.exp_probs],
            where="post",
            ls="--",
            color=options.colors[1],
            label=source_labels.exp_conf_lower,
        )
        ax.step(
            np.r_[left, raw.exp_conf_upper],
            np.r_[0.0, raw.exp_probs],
            where="post",
            ls=":",
            color=options.colors[1],
            label=source_labels.exp_conf_upper,
        )
        ax.set_title(f"MAVM: d+={raw.d_plus:.4g}, d-={raw.d_minus:.4g}")
    units = val_case.domain.observables[observable].units
    ax.set_xlabel(f"{observable} [{units}]")
    ax.set_ylabel("Cumulative probability")
    ax.set_ylim(0.0, 1.02)
    ax.grid(alpha=0.3)
    ax.legend()
    _apply_plot_options(fig, ax, plot_opts, labels)
    return fig, ax


def plot_val_cdf_envelope(
    val_case: ValCase,
    observable: str,
    *,
    source: EValSource = EValSource.SIM,
    point: int = 0,
    component: int = 0,
    time: int = 0,
    epis_trans: int | None = None,
    epis_nuis: int | None = None,
    offset: EEpisOffset | None = None,
    plot_opts: PlotOptsGeneral | None = None,
    labels: ValPlotLabels | None = None,
    ax: plt.Axes | None = None,
) -> tuple[plt.Figure, plt.Axes]:
    """Plot pointwise envelope of complete conditional CDF scenarios."""
    if not isinstance(source, EValSource):
        raise TypeError("Source must be EValSource.SIM or EValSource.EXP.")
    quantity = getattr(val_case, source).quantities[observable]
    support, lower, upper = extract_ecdf_envelope(
        quantity,
        point=point,
        component=component,
        time=time,
        epis_trans=epis_trans,
        epis_nuis=epis_nuis,
        offset=offset,
    )
    margin = max((support[-1] - support[0]) * 0.02, 1.0e-12)
    support = np.r_[support[0] - margin, support]
    lower = np.r_[0.0, lower]
    upper = np.r_[0.0, upper]
    fig, ax = _create_axes(ax, plot_opts)
    source_labels = ValPlotLabels() if labels is None else labels
    source_label = (
        source_labels.simulation
        if source is EValSource.SIM
        else source_labels.experiment
    )
    ax.step(
        support,
        lower,
        where="post",
        label=f"{source_label}: pointwise lower",
    )
    ax.step(
        support,
        upper,
        where="post",
        label=f"{source_label}: pointwise upper",
    )
    ax.fill_between(support, lower, upper, step="post", alpha=0.2)
    units = val_case.domain.observables[observable].units
    ax.set_xlabel(f"{observable} [{units}]")
    ax.set_ylabel("Cumulative probability")
    ax.set_ylim(0.0, 1.02)
    ax.grid(alpha=0.3)
    ax.legend()
    _apply_plot_options(fig, ax, plot_opts, labels)
    return fig, ax


def _select_field(
    field: ValMetricField,
    selectors: dict[EValResAxis, int],
    keep: EValResAxis,
    values: np.ndarray | None = None,
) -> np.ndarray:
    if keep not in field.axes:
        raise ValueError(f"Result has no {keep} axis.")
    array = field.metric if values is None else values
    indices: list[int | slice] = []
    for axis, size in zip(field.axes, array.shape, strict=True):
        if axis == keep:
            indices.append(slice(None))
        elif axis in selectors:
            indices.append(selectors[axis])
        elif size == 1:
            indices.append(0)
        else:
            raise ValueError(f"Select a {axis} index before plotting.")
    return array[tuple(indices)]


def plot_val_metric_map(
    result: ValResult,
    observable: str,
    metric: EMetricType,
    *,
    selectors: dict[EValResAxis, int] | None = None,
    direction: str = "metric",
    plot_opts: PlotOptsGeneral | None = None,
    labels: ValPlotLabels | None = None,
    ax: plt.Axes | None = None,
) -> tuple[plt.Figure, plt.Axes]:
    """Scatter a metric field over its registered 2D point coordinates."""
    field = result.get(observable, metric)
    values = getattr(field, direction)
    if values is None:
        raise ValueError(f"Metric has no {direction} field.")
    selected = _select_field(field, selectors or {}, EValResAxis.POINT, values)
    coords = result.domain.observables[observable].geometry.coords
    if coords is None:
        raise ValueError("Metric map needs spatial point coordinates.")
    fig, ax = _create_axes(ax, plot_opts)
    plot = ax.scatter(
        coords[:, 0],
        coords[:, 1],
        c=selected,
        cmap=(plot_opts or PlotOptsGeneral()).cmap_seq,
        s=65,
    )
    ax.set_aspect("equal")
    coordinate_units = result.domain.observables[observable].geometry.coordinate_units
    suffix = f" [{coordinate_units}]" if coordinate_units else ""
    ax.set_xlabel(f"x{suffix}")
    ax.set_ylabel(f"y{suffix}")
    ax.set_title(f"{observable}: {_metric_label(metric, direction)}")
    colorbar = fig.colorbar(
        plot,
        ax=ax,
        label=(
            labels.colorbar_label
            if labels is not None and labels.colorbar_label is not None
            else (
                f"{_metric_label(metric, direction)} "
                f"[{_metric_units(result, observable, metric)}]"
            )
        ),
    )
    options = PlotOptsGeneral() if plot_opts is None else plot_opts
    colorbar.ax.tick_params(labelsize=options.font_tick_size)
    colorbar.set_label(
        colorbar.ax.get_ylabel(),
        fontsize=options.font_ax_size,
        fontname=options.font_name,
    )
    _apply_plot_options(fig, ax, plot_opts, labels)
    return fig, ax


def plot_val_metric_time(
    result: ValResult,
    observable: str,
    metric: EMetricType,
    *,
    selectors: dict[EValResAxis, int] | None = None,
    direction: str = "metric",
    plot_opts: PlotOptsGeneral | None = None,
    labels: ValPlotLabels | None = None,
    ax: plt.Axes | None = None,
) -> tuple[plt.Figure, plt.Axes]:
    """Plot one selected metric scenario against physical time."""
    field = result.get(observable, metric)
    values = getattr(field, direction)
    if values is None:
        raise ValueError(f"Metric has no {direction} field.")
    selected = _select_field(field, selectors or {}, EValResAxis.TIME, values)
    times = result.domain.observables[observable].times
    if times is None:
        raise ValueError("Time plot requires physical time coordinates.")
    fig, ax = _create_axes(ax, plot_opts)
    ax.plot(times, selected, marker="o")
    time_units = result.domain.observables[observable].time_units
    ax.set_xlabel(f"Time [{time_units}]" if time_units else "Time")
    ax.set_ylabel(
        f"{_metric_label(metric, direction)} "
        f"[{_metric_units(result, observable, metric)}]"
    )
    ax.set_title(observable)
    ax.grid(alpha=0.3)
    _apply_plot_options(fig, ax, plot_opts, labels)
    return fig, ax


def plot_val_metric_distribution(
    result: ValResult,
    observable: str,
    metric: EMetricType,
    *,
    direction: str = "metric",
    selectors: dict[EValResAxis, int] | None = None,
    plot_opts: PlotOptsGeneral | None = None,
    labels: ValPlotLabels | None = None,
    ax: plt.Axes | None = None,
) -> tuple[plt.Figure, plt.Axes]:
    """Histogram finite metric values over selected physical/scenario cells."""
    field = result.get(observable, metric)
    index = tuple((selectors or {}).get(axis, slice(None)) for axis in field.axes)
    selected_field = getattr(field, direction)
    if selected_field is None:
        raise ValueError(f"Metric has no {direction} field.")
    values = selected_field[index].ravel()
    values = values[np.isfinite(values)]
    if not len(values):
        raise ValueError("No finite metric values selected.")
    fig, ax = _create_axes(ax, plot_opts)
    ax.hist(values, bins=min(20, len(values)))
    ax.set_xlabel(
        f"{_metric_label(metric, direction)} "
        f"[{_metric_units(result, observable, metric)}]"
    )
    ax.set_ylabel("Cell count")
    ax.set_title(observable)
    _apply_plot_options(fig, ax, plot_opts, labels)
    return fig, ax


def _create_point_axes(
    points: tuple[int, ...],
    columns: int,
    plot_opts: PlotOptsGeneral | None,
) -> tuple[plt.Figure, np.ndarray]:
    if not points or columns < 1:
        raise ValueError("Supply at least one point and a positive column count.")
    options = PlotOptsGeneral() if plot_opts is None else plot_opts
    columns = min(columns, len(points))
    rows = (len(points) + columns - 1) // columns
    width, height = options.single_fig_size_landscape
    fig, axes = plt.subplots(
        rows,
        columns,
        squeeze=False,
        figsize=(width * columns, height * rows),
        layout="constrained",
    )
    for ax in axes.flat[len(points) :]:
        ax.set_visible(False)
    return fig, axes


def plot_val_cdfs_points(
    val_case: ValCase,
    result: ValResult,
    observable: str,
    *,
    points: tuple[int, ...],
    component: int = 0,
    time: int = 0,
    sim_trans: int = 0,
    sim_nuis: int = 0,
    exp_trans: int = 0,
    exp_nuis: int = 0,
    sim_offset: EEpisOffset = EEpisOffset.NOMINAL,
    exp_offset: EEpisOffset = EEpisOffset.NOMINAL,
    metric: MetricMAVM | None = None,
    columns: int = 3,
    plot_opts: PlotOptsGeneral | None = None,
    labels: ValPlotLabels | None = None,
) -> tuple[plt.Figure, np.ndarray]:
    """Compare simulation and experiment at each point in a subplot grid.

    The returned axes array always has shape (rows, columns). Unused panels are
    hidden. Each panel identifies its point, component and physical time.
    """
    fig, axes = _create_point_axes(points, columns, plot_opts)
    observable_data = val_case.domain.observables[observable]
    time_label = (
        "steady"
        if observable_data.times is None
        else (f"t={observable_data.times[time]:g} {observable_data.time_units}")
    )
    for point, ax in zip(points, axes.flat, strict=False):
        plot_val_cdfs(
            val_case,
            result,
            observable,
            point=point,
            component=component,
            time=time,
            sim_trans=sim_trans,
            sim_nuis=sim_nuis,
            exp_trans=exp_trans,
            exp_nuis=exp_nuis,
            sim_offset=sim_offset,
            exp_offset=exp_offset,
            metric=metric,
            plot_opts=plot_opts,
            labels=labels,
            ax=ax,
        )
        heading = ax.get_title()
        ax.set_title(
            f"Point {point}: {observable_data.components[component]}, {time_label}"
            + (f"\n{heading}" if heading else ""),
            fontsize=(plot_opts or PlotOptsGeneral()).font_head_size,
        )
    return fig, axes


def plot_val_metric_time_points(
    result: ValResult,
    observable: str,
    metric: EMetricType,
    *,
    points: tuple[int, ...],
    selectors: dict[EValResAxis, int] | None = None,
    direction: str = "metric",
    columns: int = 3,
    plot_opts: PlotOptsGeneral | None = None,
    labels: ValPlotLabels | None = None,
) -> tuple[plt.Figure, np.ndarray]:
    """Plot the same selected scenario over time for multiple point panels."""
    fig, axes = _create_point_axes(points, columns, plot_opts)
    for point, ax in zip(points, axes.flat, strict=False):
        selected = dict(selectors or {})
        selected[EValResAxis.POINT] = point
        plot_val_metric_time(
            result,
            observable,
            metric,
            selectors=selected,
            direction=direction,
            plot_opts=plot_opts,
            labels=labels,
            ax=ax,
        )
        ax.set_title(
            f"Point {point}: {ax.get_title()}",
            fontsize=(plot_opts or PlotOptsGeneral()).font_head_size,
        )
    return fig, axes
