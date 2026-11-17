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
from pyvale.valid.metricstrategy import EMetricScale, EMetricType
from pyvale.valid.validtools import (
    _evaluate_ecdf_quantiles,
    build_metric_selection,
    calc_ecdf_view,
    create_metric_label,
    extract_metric_values,
    select_metric_output,
)
from pyvale.valid.validtypes import (
    EEpisOffset,
    EMetricOutput,
    EValidECDFBounds,
    EValidECDFCentre,
    EValidECDFReduce,
    EValidMAVMDiagnostic,
    EValidResAxis,
    EValidSource,
    MAVMCorrection,
    ValidCase,
    ValidECDF,
    ValidECDFSummary,
    ValidECDFView,
    ValidECDFViewData,
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
    if result.metric_configs[metric].get_scale() is not EMetricScale.PHYSICAL:
        return "1"
    return result.observables[observable].phys_units


def plot_sim_ecdfs(
    valid_case: ValidCase,
    observable: str,
    *,
    view: ValidECDFView | None = None,
    offset: EEpisOffset | None = EEpisOffset.NOMINAL,
    point_idxs: tuple[int, ...] | None = (0,),
    component_idx: int = 0,
    time_idx: int = 0,
    rows: int | None = None,
    columns: int | None = None,
    valid_plot_opts: ValidPlotOpts | None = None,
    ax: Axes | None = None,
) -> tuple[Figure, np.ndarray]:
    """Plot sim conditional ECDFs or epistemic summaries; return a 2D axes array.

    Default to the outer envelope over both epistemic axes at point zero.
    point_idxs=None explicitly selects every physical point. An existing ax
    accepts one point only. No metric calculation or result is required.
    """
    return _plot_ecdf_sources(
        valid_case,
        observable,
        ((EValidSource.SIM, view, offset),),
        point_idxs=point_idxs,
        component_idx=component_idx,
        time_idx=time_idx,
        rows=rows,
        columns=columns,
        valid_plot_opts=valid_plot_opts,
        ax=ax,
    )


def plot_exp_ecdfs(
    valid_case: ValidCase,
    observable: str,
    *,
    view: ValidECDFView | None = None,
    offset: EEpisOffset | None = EEpisOffset.NOMINAL,
    point_idxs: tuple[int, ...] | None = (0,),
    component_idx: int = 0,
    time_idx: int = 0,
    rows: int | None = None,
    columns: int | None = None,
    valid_plot_opts: ValidPlotOpts | None = None,
    ax: Axes | None = None,
) -> tuple[Figure, np.ndarray]:
    """Plot exp ECDFs with independently selected epistemic summaries and offsets."""
    return _plot_ecdf_sources(
        valid_case,
        observable,
        ((EValidSource.EXP, view, offset),),
        point_idxs=point_idxs,
        component_idx=component_idx,
        time_idx=time_idx,
        rows=rows,
        columns=columns,
        valid_plot_opts=valid_plot_opts,
        ax=ax,
    )


def plot_sim_exp_cdfs(
    valid_case: ValidCase,
    observable: str,
    *,
    sim_view: ValidECDFView | None = None,
    exp_view: ValidECDFView | None = None,
    sim_offset: EEpisOffset | None = EEpisOffset.NOMINAL,
    exp_offset: EEpisOffset | None = EEpisOffset.NOMINAL,
    point_idxs: tuple[int, ...] | None = (0,),
    component_idx: int = 0,
    time_idx: int = 0,
    rows: int | None = None,
    columns: int | None = None,
    valid_plot_opts: ValidPlotOpts | None = None,
    ax: Axes | None = None,
) -> tuple[Figure, np.ndarray]:
    """Compare sim and exp families without pooling or selecting a hidden scenario.

    Each source defaults to a shaded pointwise epistemic envelope, or one ECDF
    when only one scenario remains. Epistemic envelopes are not confidence
    intervals. Keep MAVM confidence/support diagnostics in their dedicated plots.
    """
    return _plot_ecdf_sources(
        valid_case,
        observable,
        (
            (EValidSource.SIM, sim_view, sim_offset),
            (EValidSource.EXP, exp_view, exp_offset),
        ),
        point_idxs=point_idxs,
        component_idx=component_idx,
        time_idx=time_idx,
        rows=rows,
        columns=columns,
        valid_plot_opts=valid_plot_opts,
        ax=ax,
    )


def _plot_ecdf_sources(
    valid_case: ValidCase,
    observable: str,
    sources: tuple[tuple[EValidSource, ValidECDFView | None, EEpisOffset | None], ...],
    *,
    point_idxs: tuple[int, ...] | None,
    component_idx: int,
    time_idx: int,
    rows: int | None,
    columns: int | None,
    valid_plot_opts: ValidPlotOpts | None,
    ax: Axes | None,
) -> tuple[Figure, np.ndarray]:
    observable_data = valid_case.observables[observable]
    point_count = (
        1 if observable_data.coords is None else observable_data.coords.shape[0]
    )
    selected_points = tuple(range(point_count)) if point_idxs is None else point_idxs
    if not isinstance(selected_points, tuple) or not selected_points:
        raise ValueError("Supply a nonempty tuple of point indices.")
    if any(
        isinstance(index, (bool, np.bool_)) or not isinstance(index, (int, np.integer))
        for index in selected_points
    ):
        raise TypeError("Point indices must be integers.")
    if any(index < 0 or index >= point_count for index in selected_points):
        raise IndexError("Point index is outside the observable.")
    if ax is not None and (
        len(selected_points) != 1 or rows is not None or columns is not None
    ):
        raise ValueError("An existing axes accepts one point and no grid layout.")
    # Prepare before creating any figure, so invalid selections leave no figures behind.
    prepared: list[list[tuple[EValidSource, ValidECDFView, ValidECDFViewData]]] = []
    for point in selected_points:
        point_sources = []
        for source, view, offset in sources:
            selected_view = ValidECDFView() if view is None else view
            quantity = (
                valid_case.sim if source is EValidSource.SIM else valid_case.exp
            )[observable]
            curves = calc_ecdf_view(
                quantity,
                view=selected_view,
                point_idx=point,
                component_idx=component_idx,
                time_idx=time_idx,
                offset=offset,
            )
            point_sources.append((source, selected_view, curves))
        prepared.append(point_sources)
    options = ValidPlotOpts() if valid_plot_opts is None else valid_plot_opts
    if ax is None:
        fig, axes = create_valid_plot_grid(
            len(selected_points), rows=rows, columns=columns, valid_plot_opts=options
        )
    else:
        fig, ax = _create_axes(ax, options)
        axes = np.array([[ax]], dtype=object)
    for point, point_sources, panel in zip(
        selected_points, prepared, axes.flat, strict=False
    ):
        for source, view, curves in point_sources:
            _plot_ecdf_view(panel, source, view, curves, options)
        time_label = (
            "steady"
            if observable_data.times is None
            else f"t={observable_data.times[time_idx]:g} {observable_data.time_units}"
        )
        panel.set_title(
            f"Point {point}: {observable_data.components[component_idx]}, {time_label}"
        )
        panel.set_xlabel(f"{observable} [{observable_data.phys_units}]")
        panel.set_ylabel("Cumulative probability")
        panel.set_ylim(0.0, 1.02)
        panel.grid(alpha=0.3)
        panel.legend()
        _apply_plot_options(fig, panel, options)
    return fig, axes


def _plot_ecdf_view(
    ax: Axes,
    source: EValidSource,
    view: ValidECDFView,
    curves: ValidECDFViewData,
    options: ValidPlotOpts,
) -> None:
    label = options.sim_label if source is EValidSource.SIM else options.exp_label
    color = options.sim_color if source is EValidSource.SIM else options.exp_color
    linestyle = (
        options.sim_linestyle if source is EValidSource.SIM else options.exp_linestyle
    )
    quantiles = curves.quantiles.reshape(-1, curves.probabilities.size)
    single = quantiles.shape[0] == 1
    reduction_labels = []
    for name, reduction in (("nuis.", view.nuis_reduce), ("trans.", view.trans_reduce)):
        if reduction is not EValidECDFReduce.KEEP:
            reduction_labels.append(f"{reduction.name.lower()} {name}")
    if single:
        curve_label = label
        if reduction_labels:
            curve_label += " " + " / ".join(reduction_labels) + " quantile"
        _plot_quantile_curve(
            ax,
            quantiles[0],
            curves.probabilities,
            color=color,
            label=curve_label,
            linestyle=linestyle,
        )
    else:
        if view.bounds:
            lower = np.min(quantiles, axis=0)
            upper = np.max(quantiles, axis=0)
            _plot_quantile_curve(
                ax,
                lower,
                curves.probabilities,
                color=color,
                label=f"{label} lower envelope",
                linestyle=options.lower_linestyle,
            )
            _plot_quantile_curve(
                ax,
                upper,
                curves.probabilities,
                color=color,
                label=f"{label} upper envelope",
                linestyle=options.upper_linestyle,
            )
            if options.shade_bounds and not np.array_equal(lower, upper):
                ax.fill_betweenx(
                    np.r_[0.0, curves.probabilities],
                    np.r_[lower[0], lower],
                    np.r_[upper[0], upper],
                    step="pre",
                    color=options.bounds_color or color,
                    alpha=options.bounds_alpha,
                )
        if view.centre is not EValidECDFCentre.NONE:
            centre = (
                np.mean(quantiles, axis=0)
                if view.centre is EValidECDFCentre.MEAN
                else np.median(quantiles, axis=0)
            )
            centre_label = f"{label} {view.centre.name.lower()} quantile"
            _plot_quantile_curve(
                ax,
                centre,
                curves.probabilities,
                color=color,
                label=centre_label,
                linestyle=linestyle,
            )
    # A sole original ECDF is already drawn above. Reduced single curves can
    # still be shown alongside their original selected scenario overlays.
    original_single = single and not reduction_labels and len(curves.scenarios) == 1
    if not original_single:
        for quantiles, (trans_idx, nuis_idx, offset) in zip(
            curves.scenario_quantiles, curves.scenarios, strict=True
        ):
            scenario_label = f"{label} trans. {trans_idx}, nuis. {nuis_idx}"
            if offset is not EEpisOffset.NOMINAL:
                scenario_label += f", {offset.name.lower()} offset"
            _plot_quantile_curve(
                ax,
                quantiles,
                curves.probabilities,
                color=color,
                label=scenario_label,
                linestyle=linestyle,
            )


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
