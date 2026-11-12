# ==============================================================================
# pyvale: the python validation engine
# License: MIT
# Copyright (C) 2026 The Computer Aided Validation Team
# ==============================================================================

"""Diagnostic plots for validation samples and labelled metric fields."""

from dataclasses import dataclass
from math import ceil, sqrt

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import MaxNLocator, ScalarFormatter

from pyvale.sensorsim.visualopts import PlotOptsGeneral
from pyvale.valid.metrics import calc_mavm_1d
from pyvale.valid.metricstrategy import EMetricType, MetricMAVM
from pyvale.valid.valengine import (
    EValResAxis,
    ValResult,
)
from pyvale.valid.valtools import (
    calc_ecdf,
    calc_ecdf_summary,
    create_metric_label,
    evaluate_ecdf_quantiles,
    extract_aleatory_scenarios,
    extract_ecdf_envelope,
    extract_metric_values,
    select_metric_part,
)
from pyvale.valid.valtypes import (
    EEpisOffset,
    EValECDFBounds,
    EValMAVMDiagnostic,
    EValMetricPart,
    EValSource,
    MAVMCorrection,
    ValArray,
    ValCase,
    ValECDF,
    ValECDFSummary,
)


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


def _metric_units(result: ValResult, observable: str, metric: EMetricType) -> str:
    if metric in (EMetricType.KS, EMetricType.CVM, EMetricType.RELATIVE_ERROR):
        return "1"
    return result.domain.observables[observable].phys_units


def plot_val_cdfs(
    val_case: ValCase,
    result: ValResult,
    observable: str,
    *,
    point_idx: int = 0,
    component_idx: int = 0,
    time_idx: int = 0,
    sim_trans_idx: int = 0,
    sim_nuis_idx: int = 0,
    exp_trans_idx: int = 0,
    exp_nuis_idx: int = 0,
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
    sim = val_case.sim[observable].select_aleatory(
        point_idx, component_idx, time_idx, sim_trans_idx, sim_nuis_idx, sim_offset
    )
    exp = val_case.exp[observable].select_aleatory(
        point_idx, component_idx, time_idx, exp_trans_idx, exp_nuis_idx, exp_offset
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
    units = val_case.domain.observables[observable].phys_units
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
    point_idx: int = 0,
    component_idx: int = 0,
    time_idx: int = 0,
    epis_trans_idx: int | None = None,
    epis_nuis_idx: int | None = None,
    offset: EEpisOffset | None = None,
    plot_opts: PlotOptsGeneral | None = None,
    labels: ValPlotLabels | None = None,
    ax: plt.Axes | None = None,
) -> tuple[plt.Figure, plt.Axes]:
    """Plot pointwise envelope of complete conditional CDF scenarios."""
    if not isinstance(source, EValSource):
        raise TypeError("Source must be EValSource.SIM or EValSource.EXP.")
    quantity = getattr(val_case, source)[observable]
    support, lower, upper = extract_ecdf_envelope(
        quantity,
        point_idx=point_idx,
        component_idx=component_idx,
        time_idx=time_idx,
        epis_trans_idx=epis_trans_idx,
        epis_nuis_idx=epis_nuis_idx,
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
    units = val_case.domain.observables[observable].phys_units
    ax.set_xlabel(f"{observable} [{units}]")
    ax.set_ylabel("Cumulative probability")
    ax.set_ylim(0.0, 1.02)
    ax.grid(alpha=0.3)
    ax.legend()
    _apply_plot_options(fig, ax, plot_opts, labels)
    return fig, ax


def plot_val_metric_time(
    result: ValResult,
    observable: str,
    metric: EMetricType,
    *,
    selectors: dict[EValResAxis, int] | None = None,
    part: EValMetricPart = EValMetricPart.TOTAL,
    plot_opts: PlotOptsGeneral | None = None,
    labels: ValPlotLabels | None = None,
    ax: plt.Axes | None = None,
) -> tuple[plt.Figure, plt.Axes]:
    """Plot one selected metric scenario against physical time."""
    field = result.get(observable, metric)
    selected = extract_metric_values(
        field,
        keep=EValResAxis.TIME,
        selectors=selectors,
        part=part,
    )
    times = result.domain.observables[observable].times
    if times is None:
        raise ValueError("Time plot requires physical time coordinates.")
    fig, ax = _create_axes(ax, plot_opts)
    ax.plot(times, selected, marker="o")
    time_units = result.domain.observables[observable].time_units
    ax.set_xlabel(f"Time [{time_units}]" if time_units else "Time")
    ax.set_ylabel(
        f"{create_metric_label(metric, part)} "
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
    part: EValMetricPart = EValMetricPart.TOTAL,
    selectors: dict[EValResAxis, int] | None = None,
    plot_opts: PlotOptsGeneral | None = None,
    labels: ValPlotLabels | None = None,
    ax: plt.Axes | None = None,
) -> tuple[plt.Figure, plt.Axes]:
    """Histogram finite metric values over selected physical/scenario cells."""
    field = result.get(observable, metric)
    selection_idx = tuple(
        (selectors or {}).get(axis, slice(None)) for axis in field.axes
    )
    selected_field = select_metric_part(field, part, selection_idxs=selection_idx)
    values = selected_field.ravel()
    values = values[np.isfinite(values)]
    if not len(values):
        raise ValueError("No finite metric values selected.")
    fig, ax = _create_axes(ax, plot_opts)
    ax.hist(values, bins=min(20, len(values)))
    ax.set_xlabel(
        f"{create_metric_label(metric, part)} "
        f"[{_metric_units(result, observable, metric)}]"
    )
    ax.set_ylabel("Cell count")
    ax.set_title(observable)
    _apply_plot_options(fig, ax, plot_opts, labels)
    return fig, ax


def create_val_plot_grid(
    point_count: int,
    *,
    rows: int | None = None,
    columns: int | None = None,
    plot_opts: PlotOptsGeneral | None = None,
) -> tuple[plt.Figure, np.ndarray]:
    """Create landscape diagnostic panels using standard plot options.

    Returns a two-dimensional axes array, hiding spare panels. Supply a panel
    count and optionally rows, columns or both to control the grid.
    """
    if not isinstance(point_count, int) or point_count < 1:
        raise ValueError("Supply a positive point count.")
    for count in (rows, columns):
        if count is not None and (not isinstance(count, int) or count < 1):
            raise ValueError(
                "Supply positive integer rows and columns for the point grid."
            )
    options = PlotOptsGeneral() if plot_opts is None else plot_opts
    if columns is None:
        columns = ceil(sqrt(point_count)) if rows is None else ceil(point_count / rows)
    if rows is None:
        columns = min(columns, point_count)
        rows = ceil(point_count / columns)
    if rows * columns < point_count:
        raise ValueError("Point grid has fewer panels than selected points.")
    width, height = options.single_fig_size_landscape
    fig, axes = plt.subplots(
        rows,
        columns,
        squeeze=False,
        figsize=(width * columns, height * rows),
        dpi=options.resolution,
        layout="constrained",
    )
    for ax in axes.flat[point_count:]:
        ax.set_visible(False)
    return fig, axes


def plot_val_cdf_envelope_points(
    val_case: ValCase,
    observable: str,
    *,
    point_idxs: tuple[int, ...] | None = None,
    source: EValSource = EValSource.SIM,
    component_idx: int = 0,
    time_idx: int = 0,
    epis_trans_idx: int | None = None,
    epis_nuis_idx: int | None = None,
    offset: EEpisOffset | None = None,
    rows: int | None = None,
    columns: int | None = None,
    plot_opts: PlotOptsGeneral | None = None,
    labels: ValPlotLabels | None = None,
) -> tuple[plt.Figure, np.ndarray]:
    """Plot conditional ECDF envelopes for selected physical observation points.

    ``point_idxs=None`` selects every point in domain order; an explicit tuple keeps
    its supplied order. Automatic layout uses a nearly square panel grid with
    landscape panels sized by PlotOptsGeneral. Specify rows, columns or both
    to control the layout. Axes always have shape (rows, columns), with spare
    panels hidden. Scenario selectors apply independently at every point.
    """
    if not isinstance(source, EValSource):
        raise TypeError("Source must be EValSource.SIM or EValSource.EXP.")
    observable_data = val_case.domain.observables[observable]
    coords = observable_data.geometry.coords
    point_count = 1 if coords is None else len(coords)
    selected_point_idxs = (
        tuple(range(point_count)) if point_idxs is None else point_idxs
    )
    if any(
        not isinstance(point_idx, (int, np.integer))
        or point_idx < 0
        or point_idx >= point_count
        for point_idx in selected_point_idxs
    ):
        raise ValueError("Point indices must identify points in the observable domain.")
    options = PlotOptsGeneral() if plot_opts is None else plot_opts
    fig, axes = create_val_plot_grid(
        len(selected_point_idxs), columns=columns, plot_opts=options, rows=rows
    )
    time_label = (
        "steady"
        if observable_data.times is None
        else f"t={observable_data.times[time_idx]:g} {observable_data.time_units}"
    )
    for point_idx, ax in zip(selected_point_idxs, axes.flat, strict=False):
        plot_val_cdf_envelope(
            val_case,
            observable,
            point_idx=point_idx,
            source=source,
            component_idx=component_idx,
            time_idx=time_idx,
            epis_trans_idx=epis_trans_idx,
            epis_nuis_idx=epis_nuis_idx,
            offset=offset,
            plot_opts=options,
            labels=labels,
            ax=ax,
        )
        heading = ax.get_title()
        ax.set_title(
            f"Point {point_idx}: {observable_data.components[component_idx]}, "
            f"{time_label}" + (f"\n{heading}" if heading else ""),
            fontsize=options.font_head_size,
        )
    return fig, axes


def plot_val_cdfs_points(
    val_case: ValCase,
    result: ValResult,
    observable: str,
    *,
    point_idxs: tuple[int, ...],
    component_idx: int = 0,
    time_idx: int = 0,
    sim_trans_idx: int = 0,
    sim_nuis_idx: int = 0,
    exp_trans_idx: int = 0,
    exp_nuis_idx: int = 0,
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
    fig, axes = create_val_plot_grid(
        len(point_idxs), columns=columns, plot_opts=plot_opts
    )
    observable_data = val_case.domain.observables[observable]
    time_label = (
        "steady"
        if observable_data.times is None
        else (f"t={observable_data.times[time_idx]:g} {observable_data.time_units}")
    )
    for point_idx, ax in zip(point_idxs, axes.flat, strict=False):
        plot_val_cdfs(
            val_case,
            result,
            observable,
            point_idx=point_idx,
            component_idx=component_idx,
            time_idx=time_idx,
            sim_trans_idx=sim_trans_idx,
            sim_nuis_idx=sim_nuis_idx,
            exp_trans_idx=exp_trans_idx,
            exp_nuis_idx=exp_nuis_idx,
            sim_offset=sim_offset,
            exp_offset=exp_offset,
            metric=metric,
            plot_opts=plot_opts,
            labels=labels,
            ax=ax,
        )
        heading = ax.get_title()
        ax.set_title(
            f"Point {point_idx}: {observable_data.components[component_idx]}, "
            f"{time_label}" + (f"\n{heading}" if heading else ""),
            fontsize=(plot_opts or PlotOptsGeneral()).font_head_size,
        )
    return fig, axes


def plot_val_metric_time_points(
    result: ValResult,
    observable: str,
    metric: EMetricType,
    *,
    point_idxs: tuple[int, ...],
    selectors: dict[EValResAxis, int] | None = None,
    part: EValMetricPart = EValMetricPart.TOTAL,
    columns: int = 3,
    plot_opts: PlotOptsGeneral | None = None,
    labels: ValPlotLabels | None = None,
) -> tuple[plt.Figure, np.ndarray]:
    """Plot the same selected scenario over time for multiple point panels."""
    fig, axes = create_val_plot_grid(
        len(point_idxs), columns=columns, plot_opts=plot_opts
    )
    for point_idx, ax in zip(point_idxs, axes.flat, strict=False):
        selected = dict(selectors or {})
        selected[EValResAxis.POINT] = point_idx
        plot_val_metric_time(
            result,
            observable,
            metric,
            selectors=selected,
            part=part,
            plot_opts=plot_opts,
            labels=labels,
            ax=ax,
        )
        ax.set_title(
            f"Point {point_idx}: {ax.get_title()}",
            fontsize=(plot_opts or PlotOptsGeneral()).font_head_size,
        )
    return fig, axes


def _plot_quantile_curve(
    ax: plt.Axes,
    quantiles: np.ndarray,
    probabilities: np.ndarray,
    *,
    color: str,
    label: str | None = None,
    linestyle: str = "-",
    alpha: float = 1.0,
) -> None:
    ax.step(
        np.r_[quantiles[0], quantiles],
        np.r_[0.0, probabilities],
        where="post",
        color=color,
        label=label,
        linestyle=linestyle,
        alpha=alpha,
    )


def _plot_distribution_summary(
    ax: plt.Axes,
    summary: ValECDFSummary,
    *,
    color: str,
    label: str,
) -> None:
    _plot_quantile_curve(
        ax,
        summary.nominal,
        summary.probabilities,
        color=color,
        label=f"{label}: mean quantile",
    )
    limit_label = (
        "pointwise bounds"
        if summary.bounds is EValECDFBounds.POINTWISE
        else "selected scenarios"
    )
    _plot_quantile_curve(
        ax,
        summary.lower,
        summary.probabilities,
        color=color,
        linestyle=":",
        label=f"{label}: {limit_label}",
    )
    _plot_quantile_curve(
        ax, summary.upper, summary.probabilities, color=color, linestyle=":"
    )
    ax.fill_betweenx(
        np.r_[0.0, summary.probabilities],
        np.r_[summary.lower[0], summary.lower],
        np.r_[summary.upper[0], summary.upper],
        step="pre",
        color=color,
        alpha=0.15,
    )


def plot_val_ecdf_family(
    quantity: ValArray,
    *,
    point_idx: int = 0,
    component_idx: int = 0,
    time_idx: int = 0,
    epis_trans_idx: int | None = None,
    epis_nuis_idx: int | None = None,
    offset: EEpisOffset | None = EEpisOffset.NOMINAL,
    bounds: EValECDFBounds = EValECDFBounds.POINTWISE,
    plot_opts: PlotOptsGeneral | None = None,
    labels: ValPlotLabels | None = None,
    ax: plt.Axes | None = None,
) -> tuple[plt.Figure, plt.Axes]:
    """Show all conditional ECDFs, their mean quantile curve and selected bounds."""
    samples = extract_aleatory_scenarios(
        quantity,
        point_idx=point_idx,
        component_idx=component_idx,
        time_idx=time_idx,
        epis_trans_idx=epis_trans_idx,
        epis_nuis_idx=epis_nuis_idx,
        offset=offset,
    )
    summary = calc_ecdf_summary(samples, bounds=bounds)
    fig, ax = _create_axes(ax, plot_opts)
    color = (plot_opts or PlotOptsGeneral()).colors[0]
    for selection_idx, sample in enumerate(samples):
        cdf = calc_ecdf(sample)
        _plot_quantile_curve(
            ax,
            cdf.quantiles,
            cdf.probabilities,
            color=color,
            alpha=0.25,
            label="Conditional ECDFs" if selection_idx == 0 else None,
        )
    _plot_distribution_summary(ax, summary, color="black", label="Scenario summary")
    ax.set_ylabel("Cumulative probability")
    ax.set_ylim(0.0, 1.02)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.25), ncol=2)
    _apply_plot_options(fig, ax, plot_opts, labels)
    return fig, ax


def plot_val_mavm_diagnostic(
    simulation: ValECDFSummary,
    experiment: ValECDFSummary | None = None,
    *,
    correction: MAVMCorrection | None = None,
    view: EValMAVMDiagnostic = EValMAVMDiagnostic.DISTRIBUTIONS,
    plot_opts: PlotOptsGeneral | None = None,
    labels: ValPlotLabels | None = None,
    ax: plt.Axes | None = None,
) -> tuple[plt.Figure, plt.Axes]:
    """Compare uncertainty bands, endpoint corrections or the outer MAVM support band.

    Correction views shift model support by scalar MAVM distances. The filled
    correction is not an experimental confidence interval or a pooled CDF.
    """
    if not isinstance(view, EValMAVMDiagnostic):
        raise TypeError("Diagnostic view must be EValMAVMDiagnostic.")
    if view is not EValMAVMDiagnostic.DISTRIBUTIONS and correction is None:
        raise ValueError("Selected diagnostic needs a MAVM correction.")
    fig, ax = _create_axes(ax, plot_opts)
    options = PlotOptsGeneral() if plot_opts is None else plot_opts
    source_labels = ValPlotLabels() if labels is None else labels
    _plot_distribution_summary(
        ax, simulation, color=options.colors[0], label=source_labels.simulation
    )
    if experiment is not None and view is not EValMAVMDiagnostic.MODEL_CORRECTION:
        _plot_distribution_summary(
            ax, experiment, color=options.colors[1], label=source_labels.experiment
        )
    if correction is not None and view is EValMAVMDiagnostic.COMPARISONS:
        for comparison, color, label in (
            (correction.upper, "tab:green", "Upper-support case"),
            (correction.lower, "tab:red", "Lower-support case"),
        ):
            _plot_quantile_curve(
                ax,
                comparison.model_quantiles + comparison.d_plus,
                comparison.model_probs,
                color=color,
                linestyle="--",
                label=f"{label}: +d+",
            )
            _plot_quantile_curve(
                ax,
                comparison.model_quantiles - comparison.d_minus,
                comparison.model_probs,
                color=color,
                linestyle=":",
                label=f"{label}: −d−",
            )
    if correction is not None and view in (
        EValMAVMDiagnostic.CORRECTION,
        EValMAVMDiagnostic.MODEL_CORRECTION,
    ):
        lower_cdf = ValECDF(
            correction.lower.model_quantiles - correction.lower.d_minus,
            correction.lower.model_probs,
        )
        upper_cdf = ValECDF(
            correction.upper.model_quantiles + correction.upper.d_plus,
            correction.upper.model_probs,
        )
        probabilities = np.unique(
            np.r_[lower_cdf.probabilities, upper_cdf.probabilities]
        )
        lower = evaluate_ecdf_quantiles(lower_cdf, probabilities)
        upper = evaluate_ecdf_quantiles(upper_cdf, probabilities)
        ax.fill_betweenx(
            np.r_[0.0, probabilities],
            np.r_[lower[0], lower],
            np.r_[upper[0], upper],
            step="pre",
            color="grey",
            alpha=0.25,
            label="Outer MAVM support-shift band",
        )
        _plot_quantile_curve(
            ax,
            lower,
            probabilities,
            color="black",
            linestyle="--",
            label=f"Model lower − d− ({correction.lower.d_minus:.4g})",
        )
        _plot_quantile_curve(
            ax,
            upper,
            probabilities,
            color="black",
            linestyle=":",
            label=f"Model upper + d+ ({correction.upper.d_plus:.4g})",
        )
    ax.set_ylabel("Cumulative probability")
    ax.set_ylim(0.0, 1.02)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.25), ncol=2)
    _apply_plot_options(fig, ax, plot_opts, labels)
    return fig, ax
