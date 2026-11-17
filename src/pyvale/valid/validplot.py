# ==============================================================================
# pyvale: the python validation engine
# License: MIT
# Copyright (C) 2026 The Computer Aided Validation Team
# ==============================================================================

"""Diagnostic plots for validation samples and labelled metric fields."""

from dataclasses import dataclass, field
from math import ceil, sqrt

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.ticker import MaxNLocator, ScalarFormatter

from pyvale.sensorsim.visualopts import PlotOptsGeneral
from pyvale.valid.metrics import calc_mavm_1d
from pyvale.valid.metricstrategy import EMetricType, MetricMAVM
from pyvale.valid.validtools import (
    _evaluate_ecdf_quantiles,
    build_metric_selection,
    calc_ecdf,
    calc_ecdf_summary,
    create_metric_label,
    extract_aleatory_scenarios,
    extract_ecdf_envelope,
    extract_metric_values,
    select_metric_output,
)
from pyvale.valid.validtypes import (
    EEpisOffset,
    EMetricOutput,
    EValidECDFBounds,
    EValidMAVMDiagnostic,
    EValidResAxis,
    EValidSource,
    MAVMCorrection,
    ValidArray,
    ValidCase,
    ValidECDF,
    ValidECDFSummary,
    ValidResult,
    resolve_metric_output,
)


@dataclass(slots=True)
class ValidPlotOpts:
    """Control validation figure layout, labels, source styles and ECDF shading."""

    title: str | None = None
    x_label: str | None = None
    y_label: str | None = None
    sim_label: str = "Sim."
    exp_label: str = "Exp."
    exp_conf_lower: str = "Exp. conf. lower"
    exp_conf_upper: str = "Exp. conf. upper"
    sim_color: str = "tab:blue"
    exp_color: str = "tab:orange"
    sim_linestyle: str = "-"
    exp_linestyle: str = "-"
    lower_linestyle: str = "--"
    upper_linestyle: str = ":"
    shade_bounds: bool = True
    bounds_alpha: float = 0.2
    bounds_color: str | None = None
    resolution: float = 300.0
    font_tick_size: float = 8.0
    font_head_size: float = 9.0
    font_ax_size: float = 8.0
    font_leg_size: float = 8.0
    font_def_weight: str = "normal"
    font_name: str = field(default_factory=lambda: PlotOptsGeneral().font_name)
    single_fig_size_landscape: tuple[float, float] = field(
        default_factory=lambda: PlotOptsGeneral().single_fig_size_landscape
    )
    lw: float = 0.8
    ms: float = 3.2


def _apply_plot_options(
    fig: Figure,
    ax: Axes,
    valid_plot_opts: ValidPlotOpts | None,
) -> None:
    options = ValidPlotOpts() if valid_plot_opts is None else valid_plot_opts
    if options.title is not None:
        ax.set_title(options.title)
    if options.x_label is not None:
        ax.set_xlabel(options.x_label)
    if options.y_label is not None:
        ax.set_ylabel(options.y_label)

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
    ax: Axes | None, valid_plot_opts: ValidPlotOpts | None
) -> tuple[Figure, Axes]:
    if ax is not None:
        figure = ax.get_figure(root=True)
        assert figure is not None
        return figure, ax
    options = ValidPlotOpts() if valid_plot_opts is None else valid_plot_opts
    return plt.subplots(figsize=options.single_fig_size_landscape, layout="constrained")


def _metric_units(result: ValidResult, observable: str, metric: EMetricType) -> str:
    if metric in (EMetricType.KS, EMetricType.CVM, EMetricType.RELATIVE_ERROR):
        return "1"
    return result.observables[observable].phys_units


def plot_valid_cdfs(
    valid_case: ValidCase,
    result: ValidResult,
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
    valid_plot_opts: ValidPlotOpts | None = None,
    ax: Axes | None = None,
) -> tuple[Figure, Axes]:
    """Show the exact conditional samples compared at one physical cell."""
    if observable not in result.metrics:
        raise KeyError(observable)
    sim = valid_case.sim[observable].select_aleatory(
        point_idx,
        component_idx,
        time_idx,
        epis_trans_idx=sim_trans_idx,
        epis_nuis_idx=sim_nuis_idx,
        offset=sim_offset,
    )
    exp = valid_case.exp[observable].select_aleatory(
        point_idx,
        component_idx,
        time_idx,
        epis_trans_idx=exp_trans_idx,
        epis_nuis_idx=exp_nuis_idx,
        offset=exp_offset,
    )
    sim = sim[np.isfinite(sim)]
    exp = exp[np.isfinite(exp)]
    if len(sim) == 0 or len(exp) == 0:
        raise ValueError("Selected cell has no finite observations.")
    raw = None
    support_min = min(np.min(sim), np.min(exp))
    support_max = max(np.max(sim), np.max(exp))
    if metric is not None:
        raw = calc_mavm_1d(
            sim, exp, alpha=metric.alpha, duplicates=metric.duplicates, tol=metric.tol
        )
        support_min = min(support_min, np.min(raw.exp_conf_lower))
        support_max = max(support_max, np.max(raw.exp_conf_upper))
    margin = max((support_max - support_min) * 0.02, 1.0e-12)
    left = support_min - margin
    fig, ax = _create_axes(ax, valid_plot_opts)
    source_labels = ValidPlotOpts() if valid_plot_opts is None else valid_plot_opts
    options = ValidPlotOpts() if valid_plot_opts is None else valid_plot_opts
    for samples, label, color, linestyle in (
        (sim, source_labels.sim_label, options.sim_color, options.sim_linestyle),
        (exp, source_labels.exp_label, options.exp_color, options.exp_linestyle),
    ):
        cdf = calc_ecdf(samples)
        ax.step(
            np.r_[left, cdf.quantiles],
            np.r_[0.0, cdf.probabilities],
            where="post",
            label=label,
            color=color,
            linestyle=linestyle,
        )
    if raw is not None:
        ax.step(
            np.r_[left, raw.exp_conf_lower],
            np.r_[0.0, raw.exp_probs],
            where="post",
            ls=options.lower_linestyle,
            color=options.exp_color,
            label=source_labels.exp_conf_lower,
        )
        ax.step(
            np.r_[left, raw.exp_conf_upper],
            np.r_[0.0, raw.exp_probs],
            where="post",
            ls=options.upper_linestyle,
            color=options.exp_color,
            label=source_labels.exp_conf_upper,
        )
        if options.shade_bounds:
            ax.fill_betweenx(
                np.r_[0.0, raw.exp_probs],
                np.r_[raw.exp_conf_lower[0], raw.exp_conf_lower],
                np.r_[raw.exp_conf_upper[0], raw.exp_conf_upper],
                step="pre",
                color=options.bounds_color or options.exp_color,
                alpha=options.bounds_alpha,
            )
        ax.set_title(f"MAVM: d+={raw.d_plus:.4g}, d-={raw.d_minus:.4g}")
    units = valid_case.observables[observable].phys_units
    ax.set_xlabel(f"{observable} [{units}]")
    ax.set_ylabel("Cumulative probability")
    ax.set_ylim(0.0, 1.02)
    ax.grid(alpha=0.3)
    ax.legend()
    _apply_plot_options(fig, ax, valid_plot_opts)
    return fig, ax


def plot_valid_cdf_envelope(
    valid_case: ValidCase,
    observable: str,
    *,
    source: EValidSource = EValidSource.SIM,
    point_idx: int = 0,
    component_idx: int = 0,
    time_idx: int = 0,
    epis_trans_idx: int | None = None,
    epis_nuis_idx: int | None = None,
    offset: EEpisOffset | None = None,
    valid_plot_opts: ValidPlotOpts | None = None,
    ax: Axes | None = None,
) -> tuple[Figure, Axes]:
    """Plot pointwise envelope of complete conditional CDF scenarios."""
    if not isinstance(source, EValidSource):
        raise TypeError("Source must be EValidSource.SIM or EValidSource.EXP.")
    quantity = (valid_case.sim if source is EValidSource.SIM else valid_case.exp)[
        observable
    ]
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
    fig, ax = _create_axes(ax, valid_plot_opts)
    source_labels = ValidPlotOpts() if valid_plot_opts is None else valid_plot_opts
    options = source_labels
    color = options.sim_color if source is EValidSource.SIM else options.exp_color
    linestyle = (
        options.sim_linestyle if source is EValidSource.SIM else options.exp_linestyle
    )
    source_label = (
        source_labels.sim_label
        if source is EValidSource.SIM
        else source_labels.exp_label
    )
    ax.step(
        support,
        lower,
        where="post",
        label=f"{source_label}: pointwise lower",
        color=color,
        linestyle=linestyle,
    )
    ax.step(
        support,
        upper,
        where="post",
        label=f"{source_label}: pointwise upper",
        color=color,
        linestyle=linestyle,
    )
    if options.shade_bounds:
        ax.fill_between(
            support,
            lower,
            upper,
            step="post",
            color=options.bounds_color or color,
            alpha=options.bounds_alpha,
        )
    units = valid_case.observables[observable].phys_units
    ax.set_xlabel(f"{observable} [{units}]")
    ax.set_ylabel("Cumulative probability")
    ax.set_ylim(0.0, 1.02)
    ax.grid(alpha=0.3)
    ax.legend()
    _apply_plot_options(fig, ax, valid_plot_opts)
    return fig, ax


def plot_valid_metric_time(
    result: ValidResult,
    observable: str,
    metric: EMetricType,
    *,
    selectors: dict[EValidResAxis, int] | None = None,
    output: EMetricOutput | None = None,
    valid_plot_opts: ValidPlotOpts | None = None,
    ax: Axes | None = None,
) -> tuple[Figure, Axes]:
    """Plot one selected metric scenario against physical time."""
    field = result.metrics[observable][metric]
    output = resolve_metric_output(field, output)
    selected = extract_metric_values(
        field,
        keep=EValidResAxis.TIME,
        selectors=selectors,
        output=output,
    )
    times = result.observables[observable].times
    if times is None:
        raise ValueError("Time plot requires physical time coordinates.")
    fig, ax = _create_axes(ax, valid_plot_opts)
    ax.plot(times, selected, marker="o", label=create_metric_label(metric, output))
    ax.legend()
    time_units = result.observables[observable].time_units
    ax.set_xlabel(f"Time [{time_units}]" if time_units else "Time")
    ax.set_ylabel(
        f"{create_metric_label(metric, output)} "
        f"[{_metric_units(result, observable, metric)}]"
    )
    ax.set_title(observable)
    ax.grid(alpha=0.3)
    _apply_plot_options(fig, ax, valid_plot_opts)
    return fig, ax


def plot_valid_metric_distribution(
    result: ValidResult,
    observable: str,
    metric: EMetricType,
    *,
    output: EMetricOutput | None = None,
    selectors: dict[EValidResAxis, int] | None = None,
    valid_plot_opts: ValidPlotOpts | None = None,
    ax: Axes | None = None,
) -> tuple[Figure, Axes]:
    """Histogram finite metric values over selected physical/scenario cells."""
    field = result.metrics[observable][metric]
    output = resolve_metric_output(field, output)
    selection_idx = build_metric_selection(field, selectors)
    selected_field = select_metric_output(field, output, selection_idxs=selection_idx)
    values = selected_field.ravel()
    values = values[np.isfinite(values)]
    if not len(values):
        raise ValueError("No finite metric values selected.")
    fig, ax = _create_axes(ax, valid_plot_opts)
    ax.hist(values, bins=min(20, len(values)))
    ax.set_xlabel(
        f"{create_metric_label(metric, output)} "
        f"[{_metric_units(result, observable, metric)}]"
    )
    ax.set_ylabel("Cell count")
    ax.set_title(observable)
    _apply_plot_options(fig, ax, valid_plot_opts)
    return fig, ax


def create_valid_plot_grid(
    point_count: int,
    *,
    rows: int | None = None,
    columns: int | None = None,
    valid_plot_opts: ValidPlotOpts | None = None,
) -> tuple[Figure, np.ndarray]:
    """Create landscape diagnostic panels using standard plot options.

    Returns a two-dimensional axes array, hiding spare panels. Supply a panel
    count and optionally rows, columns or both to control the grid.
    """
    if (
        isinstance(point_count, (bool, np.bool_))
        or not isinstance(point_count, (int, np.integer))
        or point_count < 1
    ):
        raise ValueError("Supply a positive point count.")
    for count in (rows, columns):
        if count is not None and (
            isinstance(count, (bool, np.bool_))
            or not isinstance(count, (int, np.integer))
            or count < 1
        ):
            raise ValueError(
                "Supply positive integer rows and columns for the point grid."
            )
    options = ValidPlotOpts() if valid_plot_opts is None else valid_plot_opts
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


def plot_valid_cdf_envelope_points(
    valid_case: ValidCase,
    observable: str,
    *,
    point_idxs: tuple[int, ...] | None = None,
    source: EValidSource = EValidSource.SIM,
    component_idx: int = 0,
    time_idx: int = 0,
    epis_trans_idx: int | None = None,
    epis_nuis_idx: int | None = None,
    offset: EEpisOffset | None = None,
    rows: int | None = None,
    columns: int | None = None,
    valid_plot_opts: ValidPlotOpts | None = None,
) -> tuple[Figure, np.ndarray]:
    """Plot conditional ECDF envelopes for selected physical observation points.

    ``point_idxs=None`` selects every point in observables order; an explicit tuple keeps
    its supplied order. Automatic layout uses a nearly square panel grid with
    landscape panels sized by ValidPlotOpts. Specify rows, columns or both
    to control the layout. Axes always have shape (rows, columns), with spare
    panels hidden. Scenario selectors apply independently at every point.
    """
    if not isinstance(source, EValidSource):
        raise TypeError("Source must be EValidSource.SIM or EValidSource.EXP.")
    observable_data = valid_case.observables[observable]
    coords = observable_data.coords
    point_count = 1 if coords is None else len(coords)
    selected_point_idxs = (
        tuple(range(point_count)) if point_idxs is None else point_idxs
    )
    if any(
        isinstance(point_idx, (bool, np.bool_))
        or not isinstance(point_idx, (int, np.integer))
        or point_idx < 0
        or point_idx >= point_count
        for point_idx in selected_point_idxs
    ):
        raise ValueError(
            "Point indices must identify points in the observable observables."
        )
    quantity = (valid_case.sim if source is EValidSource.SIM else valid_case.exp)[
        observable
    ]
    for point_idx in selected_point_idxs:
        samples = extract_aleatory_scenarios(
            quantity,
            point_idx=point_idx,
            component_idx=component_idx,
            time_idx=time_idx,
            epis_trans_idx=epis_trans_idx,
            epis_nuis_idx=epis_nuis_idx,
            offset=offset,
        )
        if any(not np.any(np.isfinite(sample)) for sample in samples):
            raise ValueError("Each ECDF scenario needs a finite observation.")
    options = ValidPlotOpts() if valid_plot_opts is None else valid_plot_opts
    fig, axes = create_valid_plot_grid(
        len(selected_point_idxs), columns=columns, valid_plot_opts=options, rows=rows
    )
    time_label = (
        "steady"
        if observable_data.times is None
        else f"t={observable_data.times[time_idx]:g} {observable_data.time_units}"
    )
    for point_idx, ax in zip(selected_point_idxs, axes.flat, strict=False):
        plot_valid_cdf_envelope(
            valid_case,
            observable,
            point_idx=point_idx,
            source=source,
            component_idx=component_idx,
            time_idx=time_idx,
            epis_trans_idx=epis_trans_idx,
            epis_nuis_idx=epis_nuis_idx,
            offset=offset,
            valid_plot_opts=options,
            ax=ax,
        )
        heading = ax.get_title()
        ax.set_title(
            f"Point {point_idx}: {observable_data.components[component_idx]}, "
            f"{time_label}" + (f"\n{heading}" if heading else ""),
            fontsize=options.font_head_size,
        )
    return fig, axes


def plot_valid_cdfs_points(
    valid_case: ValidCase,
    result: ValidResult,
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
    valid_plot_opts: ValidPlotOpts | None = None,
) -> tuple[Figure, np.ndarray]:
    """Compare simulation and experiment at each point in a subplot grid.

    The returned axes array always has shape (rows, columns). Unused panels are
    hidden. Each panel identifies its point, component and physical time.
    """
    for point_idx in point_idxs:
        for quantity, trans, nuisance, offset in (
            (valid_case.sim[observable], sim_trans_idx, sim_nuis_idx, sim_offset),
            (valid_case.exp[observable], exp_trans_idx, exp_nuis_idx, exp_offset),
        ):
            samples = quantity.select_aleatory(
                point_idx,
                component_idx,
                time_idx,
                epis_trans_idx=trans,
                epis_nuis_idx=nuisance,
                offset=offset,
            )
            if not np.any(np.isfinite(samples)):
                raise ValueError("Selected cell has no finite observations.")
    fig, axes = create_valid_plot_grid(
        len(point_idxs), columns=columns, valid_plot_opts=valid_plot_opts
    )
    observable_data = valid_case.observables[observable]
    time_label = (
        "steady"
        if observable_data.times is None
        else (f"t={observable_data.times[time_idx]:g} {observable_data.time_units}")
    )
    for point_idx, ax in zip(point_idxs, axes.flat, strict=False):
        plot_valid_cdfs(
            valid_case,
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
            valid_plot_opts=valid_plot_opts,
            ax=ax,
        )
        heading = ax.get_title()
        ax.set_title(
            f"Point {point_idx}: {observable_data.components[component_idx]}, "
            f"{time_label}" + (f"\n{heading}" if heading else ""),
            fontsize=(valid_plot_opts or ValidPlotOpts()).font_head_size,
        )
    return fig, axes


def plot_valid_metric_time_points(
    result: ValidResult,
    observable: str,
    metric: EMetricType,
    *,
    point_idxs: tuple[int, ...],
    selectors: dict[EValidResAxis, int] | None = None,
    output: EMetricOutput | None = None,
    columns: int = 3,
    valid_plot_opts: ValidPlotOpts | None = None,
) -> tuple[Figure, np.ndarray]:
    """Plot the same selected scenario over time for multiple point panels."""
    output = resolve_metric_output(result.metrics[observable][metric], output)
    if result.observables[observable].times is None:
        raise ValueError("Time plot requires physical time coordinates.")
    for point_idx in point_idxs:
        selected = dict(selectors or {})
        selected[EValidResAxis.POINT] = point_idx
        extract_metric_values(
            result.metrics[observable][metric],
            keep=EValidResAxis.TIME,
            selectors=selected,
            output=output,
        )
    fig, axes = create_valid_plot_grid(
        len(point_idxs), columns=columns, valid_plot_opts=valid_plot_opts
    )
    for point_idx, ax in zip(point_idxs, axes.flat, strict=False):
        selected = dict(selectors or {})
        selected[EValidResAxis.POINT] = point_idx
        plot_valid_metric_time(
            result,
            observable,
            metric,
            selectors=selected,
            output=output,
            valid_plot_opts=valid_plot_opts,
            ax=ax,
        )
        ax.set_title(
            f"Point {point_idx}: {ax.get_title()}",
            fontsize=(valid_plot_opts or ValidPlotOpts()).font_head_size,
        )
    return fig, axes


def _plot_quantile_curve(
    ax: Axes,
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
    ax: Axes,
    summary: ValidECDFSummary,
    *,
    color: str,
    label: str,
    options: ValidPlotOpts,
    linestyle: str = "-",
) -> None:
    _plot_quantile_curve(
        ax,
        summary.nominal,
        summary.probabilities,
        color=color,
        label=f"{label}: mean quantile",
        linestyle=linestyle,
    )
    limit_label = (
        "pointwise bounds"
        if summary.bounds is EValidECDFBounds.POINTWISE
        else "selected scenarios"
    )
    _plot_quantile_curve(
        ax,
        summary.lower,
        summary.probabilities,
        color=color,
        linestyle=options.lower_linestyle,
        label=f"{label}: {limit_label}",
    )
    _plot_quantile_curve(
        ax,
        summary.upper,
        summary.probabilities,
        color=color,
        linestyle=options.upper_linestyle,
    )
    if options.shade_bounds:
        ax.fill_betweenx(
            np.r_[0.0, summary.probabilities],
            np.r_[summary.lower[0], summary.lower],
            np.r_[summary.upper[0], summary.upper],
            step="pre",
            color=options.bounds_color or color,
            alpha=options.bounds_alpha,
        )


def plot_valid_ecdf_family(
    quantity: ValidArray,
    *,
    point_idx: int = 0,
    component_idx: int = 0,
    time_idx: int = 0,
    epis_trans_idx: int | None = None,
    epis_nuis_idx: int | None = None,
    offset: EEpisOffset | None = EEpisOffset.NOMINAL,
    bounds: EValidECDFBounds = EValidECDFBounds.POINTWISE,
    valid_plot_opts: ValidPlotOpts | None = None,
    ax: Axes | None = None,
) -> tuple[Figure, Axes]:
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
    fig, ax = _create_axes(ax, valid_plot_opts)
    color = (valid_plot_opts or ValidPlotOpts()).sim_color
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
    _plot_distribution_summary(
        ax,
        summary,
        color="black",
        label="Scenario summary",
        options=valid_plot_opts or ValidPlotOpts(),
    )
    ax.set_ylabel("Cumulative probability")
    ax.set_ylim(0.0, 1.02)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.25), ncol=2)
    _apply_plot_options(fig, ax, valid_plot_opts)
    return fig, ax


def plot_valid_mavm_diagnostic(
    sim: ValidECDFSummary,
    exp: ValidECDFSummary | None = None,
    *,
    correction: MAVMCorrection | None = None,
    view: EValidMAVMDiagnostic = EValidMAVMDiagnostic.DISTRIBUTIONS,
    valid_plot_opts: ValidPlotOpts | None = None,
    ax: Axes | None = None,
) -> tuple[Figure, Axes]:
    """Compare uncertainty bands, endpoint corrections or the outer MAVM support band.

    Correction views shift model support by scalar MAVM distances. The filled
    correction is not an experimental confidence interval or a pooled CDF.
    """
    if not isinstance(view, EValidMAVMDiagnostic):
        raise TypeError("Diagnostic view must be EValidMAVMDiagnostic.")
    if view is not EValidMAVMDiagnostic.DISTRIBUTIONS and correction is None:
        raise ValueError("Selected diagnostic needs a MAVM correction.")
    fig, ax = _create_axes(ax, valid_plot_opts)
    options = ValidPlotOpts() if valid_plot_opts is None else valid_plot_opts
    source_labels = ValidPlotOpts() if valid_plot_opts is None else valid_plot_opts
    _plot_distribution_summary(
        ax,
        sim,
        color=options.sim_color,
        label=source_labels.sim_label,
        options=options,
        linestyle=options.sim_linestyle,
    )
    if exp is not None and view is not EValidMAVMDiagnostic.MODEL_CORRECTION:
        _plot_distribution_summary(
            ax,
            exp,
            color=options.exp_color,
            label=source_labels.exp_label,
            options=options,
            linestyle=options.exp_linestyle,
        )
    if correction is not None and view is EValidMAVMDiagnostic.COMPARISONS:
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
        EValidMAVMDiagnostic.CORRECTION,
        EValidMAVMDiagnostic.MODEL_CORRECTION,
    ):
        lower_cdf = ValidECDF(
            correction.lower.model_quantiles - correction.lower.d_minus,
            correction.lower.model_probs,
        )
        upper_cdf = ValidECDF(
            correction.upper.model_quantiles + correction.upper.d_plus,
            correction.upper.model_probs,
        )
        probabilities = np.unique(
            np.r_[lower_cdf.probabilities, upper_cdf.probabilities]
        )
        lower = _evaluate_ecdf_quantiles(lower_cdf, probabilities)
        upper = _evaluate_ecdf_quantiles(upper_cdf, probabilities)
        if options.shade_bounds:
            ax.fill_betweenx(
                np.r_[0.0, probabilities],
                np.r_[lower[0], lower],
                np.r_[upper[0], upper],
                step="pre",
                color=options.bounds_color or "grey",
                alpha=options.bounds_alpha,
                label="Outer MAVM support-shift band",
            )
        _plot_quantile_curve(
            ax,
            lower,
            probabilities,
            color="black",
            linestyle="--",
            label=f"Sim. lower − d− ({correction.lower.d_minus:.4g})",
        )
        _plot_quantile_curve(
            ax,
            upper,
            probabilities,
            color="black",
            linestyle=":",
            label=f"Sim. upper + d+ ({correction.upper.d_plus:.4g})",
        )
    ax.set_ylabel("Cumulative probability")
    ax.set_ylim(0.0, 1.02)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.25), ncol=2)
    _apply_plot_options(fig, ax, valid_plot_opts)
    return fig, ax
