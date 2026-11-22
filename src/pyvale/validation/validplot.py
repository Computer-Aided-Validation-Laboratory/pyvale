# ==============================================================================
# pyvale: the python validation engine
# License: MIT
# Copyright (C) 2026 The Computer Aided Validation Team
# ==============================================================================

"""Diagnostic plots for validation samples and labelled metric fields."""

from dataclasses import dataclass, field, replace
from enum import Enum, auto
from math import ceil, sqrt

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.colors import to_rgba
from matplotlib.figure import Figure
from matplotlib.ticker import MaxNLocator, ScalarFormatter

from pyvale.sensorsim.visualopts import PlotOptsGeneral
from pyvale.validation.ecdftypes import (
    ECDFSuppBounds,
    EValidECDFCentre,
    EValidECDFReduce,
    ValidECDF,
    ValidECDFSummary,
    ValidECDFView,
    ValidECDFViewData,
)
from pyvale.validation.metrics import calc_u_pooling_1d
from pyvale.validation.metricstrategy import (
    EMetricScale,
    EMetricType,
    IValidMetric,
    MetricMAVM,
)
from pyvale.validation.validecdf import (
    _evaluate_ecdf_quantiles,
    _reduce_ecdf_quantiles,
    calc_ecdf,
    calc_ecdf_view,
)
from pyvale.validation.validprofile import _mavm_confidence_shifts
from pyvale.validation.validtools import (
    _build_metric_selection,
    _create_metric_label,
    extract_metric_values,
    select_metric_output,
)
from pyvale.validation.validtypes import (
    EEpisOffset,
    EMetricOut,
    EValidMAVMDiagnostic,
    EValidResAxis,
    EValidSource,
    MAVMBoundPairs,
    MetricField,
    ValidCase,
    ValidMetricExtrema,
    ValidResult,
)

_LineStyle = str | tuple[float, tuple[float, ...]]


@dataclass(slots=True)
class ValidPlotOpts:
    """Control validation figure layout, labels, source styles and ECDF shading."""

    title: str | None = None
    x_label: str | None = None
    y_label: str | None = None
    sim_label: str = "Sim"
    exp_label: str = "Exp"
    sim_color: str = "tab:blue"
    exp_color: str = "tab:orange"
    sim_profile_color: str = "#008b95"
    exp_profile_color: str = "#a65316"
    sim_linestyle: _LineStyle = (0, (8, 3))
    exp_linestyle: _LineStyle = "-"
    sim_bounds_linestyle: _LineStyle = "-"
    exp_bounds_linestyle: _LineStyle = "--"
    sim_profile_linestyle: _LineStyle = (0, (3, 2))
    exp_profile_linestyle: _LineStyle = (0, (2, 2))
    sim_offset_linestyle: _LineStyle = "-."
    exp_offset_linestyle: _LineStyle = "-."
    exp_conf_linestyle: _LineStyle = ":"
    shade_bounds: bool = True
    bounds_alpha: float = 0.12
    metric_color: str = "grey"
    metric_line_color: str = "black"
    metric_alpha: float = 0.25
    metric_linestyle: _LineStyle = "-."
    metric_alt_linestyle: _LineStyle = "--"
    scenario_alpha: float = 0.25
    legend_outside: bool = False
    legend_columns: int = 1
    resolution: float = 300.0
    font_tick_size: float = 8.0
    font_head_size: float = 9.0
    font_ax_size: float = 8.0
    font_leg_size: float = 8.0
    font_ecdf_leg_size: float = 6.5
    font_def_weight: str = "normal"
    font_name: str = field(default_factory=lambda: PlotOptsGeneral().font_name)
    single_fig_size_landscape: tuple[float, float] = field(
        default_factory=lambda: PlotOptsGeneral().single_fig_size_landscape
    )
    lw: float = 0.8
    ecdf_lw_scale: float = 1.5
    ms: float = 3.2


class EShowECDFBounds(Enum):
    OUTER = auto()
    ALL = auto()


class EMetricDisp(Enum):
    SUPPORT_SHIFT = auto()
    INTEGRATION_AREA = auto()
    CDF_DIFFERENCE = auto()
    ANNOTATION = auto()


@dataclass(slots=True)
class ValidECDFReference:
    """Normalise by a quantile of the displayed family's mean quantile curve."""

    source: EValidSource = EValidSource.SIM
    probability: float = 0.5


@dataclass(slots=True)
class ValidECDFComparison:
    """Choose an original conditional sim/exp pair for a metric overlay."""

    sim_trans_idx: int = 0
    sim_nuis_idx: int = 0
    exp_trans_idx: int = 0
    exp_nuis_idx: int = 0
    sim_offset: EEpisOffset = EEpisOffset.NOMINAL
    exp_offset: EEpisOffset = EEpisOffset.NOMINAL


@dataclass(slots=True)
class ECDFPlot:
    """Composable figure and the exact source selections used in each panel.

    Created by the source plotting functions. Source arrays are borrowed and
    must not be mutated while adding overlays. Axes always form a 2D grid.
    """

    figure: Figure
    axes: np.ndarray
    valid_case: ValidCase
    observable: str
    point_idxs: tuple[int, ...]
    component_idx: int
    time_idx: int
    data: tuple[dict[EValidSource, ValidECDFViewData], ...]
    options: ValidPlotOpts
    normalisation: tuple[float, ...] | None = None


def _apply_plot_options(
    fig: Figure,
    ax: Axes,
    valid_plot_opts: ValidPlotOpts | None,
    *,
    ecdf: bool = False,
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
        # Draw patterned lines over solid ones without reordering legend entries.
        line.set_zorder(2.1 if line.is_dashed() else 2.0)
        if not ecdf:
            line.set_linewidth(options.lw)
        line.set_markersize(options.ms)
    legend = ax.get_legend()
    if legend is not None:
        if options.legend_outside:
            legend = ax.legend(
                loc="upper center",
                bbox_to_anchor=(0.5, -0.3),
                ncol=options.legend_columns,
            )
        else:
            legend = ax.legend(loc="upper left", ncol=options.legend_columns)
        for text in legend.get_texts():
            text.set_fontsize(
                options.font_ecdf_leg_size if ecdf else options.font_leg_size
            )
            text.set_fontname(options.font_name)


def _create_axes(
    ax: Axes | None, valid_plot_opts: ValidPlotOpts | None
) -> tuple[Figure, Axes]:
    if ax is not None:
        figure = ax.get_figure(root=True)
        assert figure is not None
        return figure, ax
    options = ValidPlotOpts() if valid_plot_opts is None else valid_plot_opts
    return plt.subplots(
        figsize=options.single_fig_size_landscape,
        dpi=options.resolution,
        layout="constrained",
    )


def _metric_units(result: ValidResult, observable: str, metric: EMetricType) -> str:
    if result.metric_configs[metric].get_scale() is not EMetricScale.PHYSICAL:
        return "-"
    units = result.observables[observable].phys_units
    return "-" if units in ("1", "") else units


def plot_sim_ecdfs(
    valid_case: ValidCase,
    observable: str,
    *,
    view: ValidECDFView | None = None,
    offset: EEpisOffset | None = EEpisOffset.NOMINAL,
    show_bounds: EShowECDFBounds = EShowECDFBounds.OUTER,
    nuisance_profile: EValidECDFReduce | None = None,
    nuisance_profile_quantile: float | None = None,
    point_idxs: tuple[int, ...] | None = (0,),
    component_idx: int = 0,
    time_idx: int = 0,
    rows: int | None = None,
    columns: int | None = None,
    valid_plot_opts: ValidPlotOpts | None = None,
    ax: Axes | None = None,
) -> ECDFPlot:
    """Plot sim conditional ECDFs or epistemic summaries; return a composable figure.

    Default to the outer envelope over both epistemic axes at point zero.
    point_idxs=None explicitly selects every physical point. An existing ax
    accepts one point only. Set nuisance_profile to overlay a nuisance reduction
    within the selected envelope. ALL also displays original scenario curves.
    """
    return _plot_ecdf_sources(
        valid_case,
        observable,
        ((EValidSource.SIM, view, offset),),
        show_bounds=show_bounds,
        confidence_alpha=None,
        nuisance_profile=nuisance_profile,
        nuisance_profile_quantile=nuisance_profile_quantile,
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
    show_bounds: EShowECDFBounds = EShowECDFBounds.OUTER,
    confidence_alpha: float | None = None,
    nuisance_profile: EValidECDFReduce | None = None,
    nuisance_profile_quantile: float | None = None,
    point_idxs: tuple[int, ...] | None = (0,),
    component_idx: int = 0,
    time_idx: int = 0,
    rows: int | None = None,
    columns: int | None = None,
    valid_plot_opts: ValidPlotOpts | None = None,
    ax: Axes | None = None,
) -> ECDFPlot:
    """Plot exp ECDFs with optional nuisance profiles and MAVM confidence bounds.

    Use offset=None for nominal and shifted ECDFs on the same axes. OUTER shows
    the enclosing bounds, while ALL includes each selected original scenario.
    """
    return _plot_ecdf_sources(
        valid_case,
        observable,
        ((EValidSource.EXP, view, offset),),
        show_bounds=show_bounds,
        confidence_alpha=confidence_alpha,
        nuisance_profile=nuisance_profile,
        nuisance_profile_quantile=nuisance_profile_quantile,
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
    show_bounds: EShowECDFBounds = EShowECDFBounds.OUTER,
    confidence_alpha: float | None = None,
    nuisance_profile: EValidECDFReduce | None = None,
    nuisance_profile_quantile: float | None = None,
    point_idxs: tuple[int, ...] | None = (0,),
    component_idx: int = 0,
    time_idx: int = 0,
    rows: int | None = None,
    columns: int | None = None,
    valid_plot_opts: ValidPlotOpts | None = None,
    ax: Axes | None = None,
) -> ECDFPlot:
    """Compare sim and exp families without pooling or selecting a hidden scenario.

    Each source defaults to a shaded pointwise epistemic envelope, or one ECDF
    when only one scenario remains. Set nuisance_profile to overlay a pointwise
    nuisance reduction within each transferable case. QUANTILE also requires
    nuisance_profile_quantile. Use exp_offset=None or sim_offset=None to display
    nominal and coherent offsets together. Set confidence_alpha to display
    MAVM experimental confidence bounds. ALL includes individual scenario curves
    and confidence pairs. Metric overlays use the original plot selections.
    """
    return _plot_ecdf_sources(
        valid_case,
        observable,
        (
            (EValidSource.EXP, exp_view, exp_offset),
            (EValidSource.SIM, sim_view, sim_offset),
        ),
        show_bounds=show_bounds,
        confidence_alpha=confidence_alpha,
        nuisance_profile=nuisance_profile,
        nuisance_profile_quantile=nuisance_profile_quantile,
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
    show_bounds: EShowECDFBounds,
    confidence_alpha: float | None,
    nuisance_profile: EValidECDFReduce | None,
    nuisance_profile_quantile: float | None,
    point_idxs: tuple[int, ...] | None,
    component_idx: int,
    time_idx: int,
    rows: int | None,
    columns: int | None,
    valid_plot_opts: ValidPlotOpts | None,
    ax: Axes | None,
) -> ECDFPlot:
    if not isinstance(show_bounds, EShowECDFBounds):
        raise TypeError("show_bounds must be EShowECDFBounds.")
    if confidence_alpha is not None and (
        isinstance(confidence_alpha, (bool, np.bool_))
        or not isinstance(confidence_alpha, (int, float, np.integer, np.floating))
        or not 0 < confidence_alpha <= 0.5
    ):
        raise ValueError("Confidence alpha must be in (0, 0.5].")
    if nuisance_profile is not None:
        if not isinstance(nuisance_profile, EValidECDFReduce):
            raise TypeError("Nuisance profile must be EValidECDFReduce.")
        if nuisance_profile is EValidECDFReduce.KEEP:
            raise ValueError("Nuisance profile requires mean, median or quantile.")
    if nuisance_profile is EValidECDFReduce.QUANTILE:
        if (
            isinstance(nuisance_profile_quantile, (bool, np.bool_))
            or not isinstance(
                nuisance_profile_quantile, (int, float, np.integer, np.floating)
            )
            or not 0 <= nuisance_profile_quantile <= 1
        ):
            raise ValueError("Nuisance profile probability must be in [0, 1].")
    elif nuisance_profile_quantile is not None:
        raise ValueError("A nuisance profile probability requires QUANTILE reduction.")
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
    profiles: list[list[tuple[EValidSource, ValidECDFView, ValidECDFViewData]]] = []
    for point in selected_points:
        point_sources = []
        point_profiles = []
        for source, view, offset in sources:
            selected_view = ValidECDFView() if view is None else view
            if show_bounds is EShowECDFBounds.ALL:
                selected_view = replace(selected_view, scenario_idxs=None)
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
            if nuisance_profile is not None:
                if (
                    not selected_view.bounds
                    or selected_view.nuis_reduce is not EValidECDFReduce.KEEP
                    or selected_view.trans_reduce is not EValidECDFReduce.KEEP
                ):
                    raise ValueError("Nuisance profiles require an unreduced envelope.")
                if curves.quantiles.shape[1] == 1:
                    continue
                profile_view = replace(
                    selected_view,
                    nuis_reduce=nuisance_profile,
                    nuis_quantile=nuisance_profile_quantile,
                    centre=EValidECDFCentre.NONE,
                    scenario_idxs=(),
                )
                profile_quantiles = _reduce_ecdf_quantiles(
                    curves.quantiles, profile_view
                )
                profile_curves = replace(
                    curves,
                    quantiles=profile_quantiles,
                    scenarios=(),
                    scenario_quantiles=curves.scenario_quantiles[:0],
                )
                point_profiles.append((source, profile_view, profile_curves))
        prepared.append(point_sources)
        profiles.append(point_profiles)
    options = ValidPlotOpts() if valid_plot_opts is None else valid_plot_opts
    if ax is None:
        fig, axes = create_valid_plot_grid(
            len(selected_points), rows=rows, columns=columns, valid_plot_opts=options
        )
    else:
        fig, ax = _create_axes(ax, options)
        axes = np.array([[ax]], dtype=object)
    for point, point_sources, point_profiles, panel in zip(
        selected_points, prepared, profiles, axes.flat, strict=False
    ):
        profile_by_source = {source: curves for source, _, curves in point_profiles}
        for source, view, curves in point_sources:
            has_offsets = curves.quantiles.shape[2] > 1
            _plot_ecdf_view(
                panel,
                source,
                view,
                curves,
                options,
                label_suffix=(
                    " Off"
                    if has_offsets
                    else " Nuis Env"
                    if source in profile_by_source
                    else ""
                ),
                offset_bounds=has_offsets,
                show_all=show_bounds is EShowECDFBounds.ALL,
            )
            if has_offsets:
                _plot_nominal_offset_view(
                    panel, source, curves, profile_by_source.get(source), options
                )
            if source is EValidSource.EXP and confidence_alpha is not None:
                _plot_exp_confidence_bounds(
                    panel, curves, confidence_alpha, show_bounds, options
                )
        for source, view, curves in point_profiles:
            if curves.quantiles.shape[2] > 1 and show_bounds is EShowECDFBounds.OUTER:
                curves = replace(curves, quantiles=curves.quantiles[:, :, :1])
            _plot_ecdf_view(
                panel,
                source,
                view,
                curves,
                options,
                profile=True,
                show_all=show_bounds is EShowECDFBounds.ALL,
            )
        time_label = (
            "steady"
            if observable_data.times is None
            else f"t={observable_data.times[time_idx]:g} {observable_data.time_units}"
        )
        panel.set_title(
            f"Point {point}: {observable_data.components[component_idx]}, {time_label}"
        )
        units = observable_data.phys_units
        units = "-" if units in ("1", "") else units
        panel.set_xlabel(f"{observable} [{units}]")
        panel.set_ylabel("Cumulative probability")
        panel.set_ylim(0.0, 1.02)
        panel.grid(alpha=0.3)
        if panel.get_legend_handles_labels()[1]:
            panel.legend()
        _apply_plot_options(fig, panel, options, ecdf=True)
    return ECDFPlot(
        fig,
        axes,
        valid_case,
        observable,
        selected_points,
        component_idx,
        time_idx,
        tuple(
            {source: curves for source, _, curves in point_sources}
            for point_sources in prepared
        ),
        options,
    )


def _ecdf_profile_label(reduction: EValidECDFReduce, quantile: float | None) -> str:
    if reduction is EValidECDFReduce.MEDIAN:
        return "Prof-Med"
    if reduction is EValidECDFReduce.MEAN:
        return "Prof-Mean"
    if reduction is EValidECDFReduce.QUANTILE:
        assert quantile is not None
        return f"Prof-Q{100 * quantile:g}"
    return ""


def _plot_ecdf_view(
    ax: Axes,
    source: EValidSource,
    view: ValidECDFView,
    curves: ValidECDFViewData,
    options: ValidPlotOpts,
    *,
    label_suffix: str = "",
    profile: bool = False,
    offset_bounds: bool = False,
    show_all: bool = False,
) -> None:
    is_sim = source is EValidSource.SIM
    label = (options.sim_label if is_sim else options.exp_label) + label_suffix
    for axis, reduction, quantile in (
        ("Nuis", view.nuis_reduce, view.nuis_quantile),
        ("Trans", view.trans_reduce, view.trans_quantile),
    ):
        method = _ecdf_profile_label(reduction, quantile)
        if method:
            label = f"{label} {axis} {method}"
    if profile:
        color = options.sim_profile_color if is_sim else options.exp_profile_color
    else:
        color = options.sim_color if is_sim else options.exp_color
    bounds_linestyle = (
        options.sim_bounds_linestyle if is_sim else options.exp_bounds_linestyle
    )
    linestyle = (
        bounds_linestyle
        if profile
        else (options.sim_linestyle if is_sim else options.exp_linestyle)
    )
    if profile:
        bounds_linestyle = (
            options.sim_profile_linestyle if is_sim else options.exp_profile_linestyle
        )
    elif offset_bounds:
        bounds_linestyle = (
            options.sim_offset_linestyle if is_sim else options.exp_offset_linestyle
        )
    quantiles = curves.quantiles.reshape(-1, curves.probabilities.size)
    single = quantiles.shape[0] == 1
    reduced = (
        view.nuis_reduce is not EValidECDFReduce.KEEP
        or view.trans_reduce is not EValidECDFReduce.KEEP
    )
    if single:
        _plot_quantile_curve(
            ax,
            quantiles[0],
            curves.probabilities,
            color=color,
            label=label,
            linestyle=linestyle,
            linewidth=options.lw * options.ecdf_lw_scale,
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
                label=label,
                linestyle=bounds_linestyle,
                linewidth=options.lw * options.ecdf_lw_scale,
            )
            _plot_quantile_curve(
                ax,
                upper,
                curves.probabilities,
                color=color,
                linestyle=bounds_linestyle,
                linewidth=options.lw * options.ecdf_lw_scale,
            )
            if options.shade_bounds and not np.array_equal(lower, upper):
                ax.fill_betweenx(
                    np.r_[0.0, curves.probabilities],
                    np.r_[lower[0], lower],
                    np.r_[upper[0], upper],
                    step="pre",
                    color=color,
                    alpha=options.bounds_alpha,
                    zorder=0.4,
                )
        if view.centre is not EValidECDFCentre.NONE:
            centre = (
                np.mean(quantiles, axis=0)
                if view.centre is EValidECDFCentre.MEAN
                else np.median(quantiles, axis=0)
            )
            _plot_quantile_curve(
                ax,
                centre,
                curves.probabilities,
                color=color,
                label=None if view.bounds else label,
                linestyle=linestyle,
                linewidth=options.lw * options.ecdf_lw_scale,
            )
    if profile and show_all and not single:
        for profile_quantiles in quantiles:
            _plot_quantile_curve(
                ax,
                profile_quantiles,
                curves.probabilities,
                color=color,
                linestyle=bounds_linestyle,
                alpha=options.scenario_alpha,
                linewidth=options.lw * options.ecdf_lw_scale,
            )
    # A sole original ECDF is already drawn above. Reduced single curves can
    # still be shown alongside their original selected scenario overlays.
    original_single = single and not reduced and len(curves.scenarios) == 1
    if not original_single:
        for scenario_index, (quantiles, (trans_idx, nuis_idx, offset)) in enumerate(
            zip(curves.scenario_quantiles, curves.scenarios, strict=True)
        ):
            scenario_style = linestyle
            if offset is not EEpisOffset.NOMINAL:
                scenario_style = (
                    options.sim_offset_linestyle
                    if is_sim
                    else options.exp_offset_linestyle
                )
            scenario_label = f"{label} Trans {trans_idx}, Nuis {nuis_idx}"
            if offset is not EEpisOffset.NOMINAL:
                scenario_label += f", {offset.name.lower()} offset"
            if (
                scenario_index == 0
                and not single
                and not view.bounds
                and view.centre is EValidECDFCentre.NONE
            ):
                scenario_label = label
            elif view.scenario_idxs is None:
                scenario_label = None
            _plot_quantile_curve(
                ax,
                quantiles,
                curves.probabilities,
                color=color,
                label=scenario_label,
                linestyle=scenario_style,
                alpha=options.scenario_alpha,
                linewidth=options.lw * options.ecdf_lw_scale,
            )


def _plot_nominal_offset_view(
    ax: Axes,
    source: EValidSource,
    curves: ValidECDFViewData,
    profile: ValidECDFViewData | None,
    options: ValidPlotOpts,
) -> None:
    # Expanded offset order follows EEpisOffset, starting with NOMINAL.
    nominal = curves.quantiles[:, :, 0].reshape(-1, curves.probabilities.size)
    is_sim = source is EValidSource.SIM
    color = options.sim_color if is_sim else options.exp_color
    label = options.sim_label if is_sim else options.exp_label
    if nominal.shape[0] > 1:
        _plot_ecdf_limits(
            ax,
            nominal,
            curves.probabilities,
            color,
            options.sim_bounds_linestyle if is_sim else options.exp_bounds_linestyle,
            f"{label} Epi Env",
            linewidth=options.lw * options.ecdf_lw_scale,
        )
    centre_curves = (
        nominal
        if profile is None
        else profile.quantiles[:, :, 0].reshape(-1, curves.probabilities.size)
    )
    _plot_quantile_curve(
        ax,
        np.mean(centre_curves, axis=0),
        curves.probabilities,
        color=color,
        linestyle=options.sim_linestyle if is_sim else options.exp_linestyle,
        label=f"{label} Nom" if nominal.shape[0] == 1 else f"{label} Nom Ctr",
        linewidth=options.lw * options.ecdf_lw_scale,
    )


def _plot_ecdf_limits(
    ax: Axes,
    quantiles: np.ndarray,
    probabilities: np.ndarray,
    color: str,
    linestyle: _LineStyle,
    label: str,
    *,
    linewidth: float,
) -> None:
    lower = np.min(quantiles, axis=0)
    upper = np.max(quantiles, axis=0)
    _plot_quantile_curve(
        ax,
        lower,
        probabilities,
        color=color,
        linestyle=linestyle,
        label=label,
        linewidth=linewidth,
    )
    if not np.array_equal(lower, upper):
        _plot_quantile_curve(
            ax,
            upper,
            probabilities,
            color=color,
            linestyle=linestyle,
            linewidth=linewidth,
        )


def _plot_exp_confidence_bounds(
    ax: Axes,
    curves: ValidECDFViewData,
    alpha: float,
    show_bounds: EShowECDFBounds,
    options: ValidPlotOpts,
) -> None:
    if alpha == 0.5:
        return
    shifts = _mavm_confidence_shifts(curves.samples, alpha)[:, None]
    if not np.any(shifts > 0):
        return
    lower = curves.original_quantiles - shifts
    upper = curves.original_quantiles + shifts
    _plot_ecdf_limits(
        ax,
        np.concatenate((lower, upper)),
        curves.probabilities,
        options.exp_color,
        options.exp_conf_linestyle,
        f"{options.exp_label} Conf",
        linewidth=options.lw,
    )
    if show_bounds is EShowECDFBounds.ALL and len(lower) > 1:
        for quantiles in np.concatenate((lower, upper)):
            _plot_quantile_curve(
                ax,
                quantiles,
                curves.probabilities,
                color=options.exp_color,
                linestyle=options.exp_conf_linestyle,
                alpha=options.scenario_alpha,
                linewidth=options.lw,
            )


def add_support_overlay(
    plot: ECDFPlot,
    supports: ECDFSuppBounds,
    *,
    output: EMetricOut | None = None,
    selectors: dict[EValidResAxis, int] | None = None,
    label: str | None = None,
) -> None:
    """Draw precomputed bounds/profiles without recalculating validation metrics.

    Select unresolved transferable or offset axes explicitly. Physical axes
    follow the plot panels; fixed physical selectors in the numerical profile
    must match. D_PLUS/D_MINUS show the corresponding extension separately.
    Omit output for the full support envelope. Bounds-based results with no
    physical axes are for a single-panel plot.
    """
    if output not in (
        None,
        EMetricOut.D_PLUS,
        EMetricOut.D_MINUS,
        EMetricOut.DISTANCE,
    ):
        raise ValueError("Support overlays require distance or directional outputs.")
    selected = selectors or {}
    if any(axis not in supports.axes for axis in selected):
        raise ValueError("Support selectors must name retained support axes.")
    if not supports.axes and not supports.selectors and len(plot.point_idxs) != 1:
        raise ValueError("Bounds without physical metadata require a single panel.")
    prepared = []
    for point in plot.point_idxs:
        physical = {
            EValidResAxis.POINT: point,
            EValidResAxis.COMPONENT: plot.component_idx,
            EValidResAxis.TIME: plot.time_idx,
        }
        for axis, index in supports.selectors.items():
            if axis in physical and physical[axis] != index:
                raise ValueError(
                    "The profile and plot select different physical cells."
                )
        indices = []
        for axis, size in zip(supports.axes, supports.lower.shape[:-1], strict=True):
            if axis in selected:
                index = selected[axis]
                if axis in physical and physical[axis] != index:
                    raise ValueError(
                        "Support selectors must match the plotted physical cell."
                    )
            elif axis in physical:
                index = physical[axis]
            elif size == 1:
                index = 0
            else:
                raise ValueError(
                    f"Select a retained {axis.value} index for this overlay."
                )
            if isinstance(index, (bool, np.bool_)) or not isinstance(
                index, (int, np.integer)
            ):
                raise TypeError("Support selector indices must be integers.")
            if not 0 <= index < size:
                raise IndexError("Support selector is outside the retained axis.")
            indices.append(index)
        selection = tuple(indices)
        lower = (
            supports.sim_upper[selection]
            if output is EMetricOut.D_PLUS
            else supports.lower[selection]
        )
        upper = (
            supports.sim_lower[selection]
            if output is EMetricOut.D_MINUS
            else supports.upper[selection]
        )
        if not np.all(np.isfinite(lower)) or not np.all(np.isfinite(upper)):
            raise ValueError("The selected support profile has no finite comparison.")
        prepared.append((lower, upper))
    options = plot.options
    if label is None:
        if supports.metric is EMetricType.MAVM and output is None:
            label = "MAVM supp"
        else:
            label = supports.metric.name if supports.metric is not None else "Support"
    for ax, (lower, upper) in zip(plot.axes.flat, prepared, strict=False):
        if np.array_equal(lower, upper):
            continue
        ax.fill_betweenx(
            np.r_[0.0, supports.probabilities],
            np.r_[lower[0], lower],
            np.r_[upper[0], upper],
            step="pre",
            color=options.metric_color,
            alpha=options.metric_alpha,
            zorder=0.5,
            label=label,
        )
        _plot_quantile_curve(
            ax,
            lower,
            supports.probabilities,
            color=options.metric_line_color,
            linestyle=options.metric_linestyle,
            linewidth=options.lw,
        )
        _plot_quantile_curve(
            ax,
            upper,
            supports.probabilities,
            color=options.metric_line_color,
            linestyle=options.metric_linestyle,
            linewidth=options.lw,
        )
        ax.legend()
        _apply_plot_options(plot.figure, ax, options, ecdf=True)


def add_normalised_ecdf_axis(
    plot: ECDFPlot,
    reference: float | ValidECDFReference,
) -> None:
    """Add a secondary x/c axis using one constant per physical panel.

    A reference quantile is evaluated on the displayed family's mean quantile
    curve, after its selected epistemic reductions. This is a constant scaling,
    not a probability-dependent displacement. Metric values retain physical units.
    """
    if plot.normalisation is not None:
        raise ValueError("This plot already has a normalised horizontal axis.")
    constants = []
    if isinstance(reference, ValidECDFReference):
        if not isinstance(reference.source, EValidSource):
            raise TypeError("Reference source must be EValidSource.")
        probability = reference.probability
        if (
            isinstance(probability, (bool, np.bool_))
            or not isinstance(probability, (int, float, np.integer, np.floating))
            or not np.isfinite(probability)
            or not 0 < probability <= 1
        ):
            raise ValueError("Reference probability must be in (0, 1].")
        for data in plot.data:
            if reference.source not in data:
                raise ValueError("The reference source is not displayed in this plot.")
            curves = data[reference.source]
            mean_quantiles = np.mean(curves.quantiles, axis=(0, 1, 2))
            index = np.searchsorted(curves.probabilities, probability, side="left")
            constants.append(float(mean_quantiles[index]))
    else:
        if isinstance(reference, (bool, np.bool_)) or not isinstance(
            reference, (int, float, np.integer, np.floating)
        ):
            raise TypeError("Reference must be a constant or ValidECDFReference.")
        constants = [float(reference)] * len(plot.point_idxs)
    if any(not np.isfinite(constant) or constant == 0 for constant in constants):
        raise ValueError("Normalisation constants must be finite and nonzero.")
    for ax, constant in zip(plot.axes.flat, constants, strict=False):
        secondary = ax.secondary_xaxis(
            "top",
            functions=(
                lambda positions, scale=constant: positions / scale,
                lambda positions, scale=constant: positions * scale,
            ),
        )
        secondary.set_xlabel(
            f"{plot.observable} norm. [-]", fontsize=plot.options.font_ax_size
        )
        secondary.xaxis.label.set_fontname(plot.options.font_name)
        secondary.tick_params(labelsize=plot.options.font_tick_size)
    plot.normalisation = tuple(constants)


def add_metric_overlay(
    plot: ECDFPlot,
    metric: IValidMetric,
    *,
    output: EMetricOut | None = None,
    display: EMetricDisp | None = None,
    comparison: ValidECDFComparison | None = None,
) -> None:
    """Compose a metric illustration with the exact selected conditional samples.

    Support shifts default to outer limits over original sim/exp comparisons,
    including when the displayed ECDF view contains mean/median summaries.
    MAVM shows the full support envelope unless a direction is selected.
    They never shift an envelope by a global maximum distance. Integration areas
    require a sole original pair or an explicit comparison. KS defaults to two
    vertical distances between the displayed lower/upper support envelopes.
    CvM and paired metrics default to annotations of original comparison values.
    No metric is calculated from synthetic mean/median observations.
    """
    kind = metric.get_kind()
    outputs = metric.get_outputs()
    if display is None:
        if kind in (EMetricType.AVM, EMetricType.MAVM):
            display = EMetricDisp.SUPPORT_SHIFT
        elif kind is EMetricType.KS:
            display = EMetricDisp.CDF_DIFFERENCE
        else:
            display = EMetricDisp.ANNOTATION
    if not isinstance(display, EMetricDisp):
        raise TypeError("Metric display must be EMetricDisp.")
    if display in (
        EMetricDisp.SUPPORT_SHIFT,
        EMetricDisp.INTEGRATION_AREA,
    ) and kind not in (EMetricType.AVM, EMetricType.MAVM):
        raise ValueError("Support shifts and integration areas require AVM or MAVM.")
    if display is EMetricDisp.CDF_DIFFERENCE and kind is not EMetricType.KS:
        raise ValueError("Vertical CDF differences require KS.")
    full_support = (
        kind is EMetricType.MAVM
        and display is EMetricDisp.SUPPORT_SHIFT
        and output is None
    )
    if full_support:
        output_indices = (
            outputs.index(EMetricOut.D_PLUS),
            outputs.index(EMetricOut.D_MINUS),
        )
    else:
        if output is None:
            if len(outputs) != 1:
                raise ValueError("Select D_PLUS or D_MINUS for this MAVM overlay.")
            output = outputs[0]
        if output not in outputs:
            raise ValueError("The selected metric does not declare that output.")
        output_indices = (outputs.index(output),)
    if comparison is not None:
        if not isinstance(comparison, ValidECDFComparison):
            raise TypeError("Comparison must be ValidECDFComparison.")
        for index in (
            comparison.sim_trans_idx,
            comparison.sim_nuis_idx,
            comparison.exp_trans_idx,
            comparison.exp_nuis_idx,
        ):
            if isinstance(index, (bool, np.bool_)) or not isinstance(
                index, (int, np.integer)
            ):
                raise TypeError("Comparison scenario indices must be integers.")
        if not isinstance(comparison.sim_offset, EEpisOffset) or not isinstance(
            comparison.exp_offset, EEpisOffset
        ):
            raise TypeError("Comparison offsets must be EEpisOffset.")
    prepared = []
    for data in plot.data:
        if EValidSource.SIM not in data or EValidSource.EXP not in data:
            raise ValueError(
                "Metric overlays require plot_sim_exp_cdfs with both sources."
            )
        sim = data[EValidSource.SIM]
        exp = data[EValidSource.EXP]
        sim_indices = tuple(range(len(sim.samples)))
        exp_indices = tuple(range(len(exp.samples)))
        if comparison is not None:
            sim_key = (
                comparison.sim_trans_idx,
                comparison.sim_nuis_idx,
                comparison.sim_offset,
            )
            exp_key = (
                comparison.exp_trans_idx,
                comparison.exp_nuis_idx,
                comparison.exp_offset,
            )
            if (
                sim_key not in sim.original_scenarios
                or exp_key not in exp.original_scenarios
            ):
                raise ValueError(
                    "Comparison is outside the plotted scenario selections."
                )
            sim_indices = (sim.original_scenarios.index(sim_key),)
            exp_indices = (exp.original_scenarios.index(exp_key),)
        if display is EMetricDisp.INTEGRATION_AREA and (
            len(sim_indices) != 1 or len(exp_indices) != 1
        ):
            raise ValueError(
                "Integration shading requires an explicit original scenario comparison."
            )
        value_columns = (
            len(output_indices)
            if display is EMetricDisp.SUPPORT_SHIFT
            else len(exp_indices)
        )
        values = np.full((len(sim_indices), value_columns), np.nan)
        # KS envelope geometry is calculated directly from displayed bounds;
        # it is not claimed to be the maximum KS over original scenario pairs.
        if display is not EMetricDisp.CDF_DIFFERENCE:
            for row, sim_index in enumerate(sim_indices):
                for column, exp_index in enumerate(exp_indices):
                    result = metric.calc(sim.samples[sim_index], exp.samples[exp_index])
                    metric_values = metric.extract_outputs(result)
                    if display is EMetricDisp.SUPPORT_SHIFT:
                        for direction_idx, output_idx in enumerate(output_indices):
                            values[row, direction_idx] = np.fmax(
                                values[row, direction_idx], metric_values[output_idx]
                            )
                    else:
                        values[row, column] = metric_values[output_indices[0]]
        prepared.append((sim, exp, sim_indices, exp_indices, values))
    options = plot.options
    metric_label = kind.name if output is None else _create_metric_label(kind, output)
    for ax, (sim, exp, sim_indices, exp_indices, values) in zip(
        plot.axes.flat, prepared, strict=False
    ):
        if display is EMetricDisp.SUPPORT_SHIFT:
            quantiles = sim.original_quantiles[list(sim_indices)]
            lower = np.min(quantiles, axis=0)
            upper = np.max(quantiles, axis=0)
            if full_support:
                upper = np.max(quantiles + values[:, 0, None], axis=0)
                lower = np.min(quantiles - values[:, 1, None], axis=0)
            elif kind is EMetricType.AVM:
                upper = np.max(quantiles + values[:, 0, None], axis=0)
                lower = np.min(quantiles - values[:, 0, None], axis=0)
            elif output is EMetricOut.D_PLUS:
                lower = upper
                upper = np.max(quantiles + values[:, 0, None], axis=0)
            else:
                upper = lower
                lower = np.min(quantiles - values[:, 0, None], axis=0)
            if np.array_equal(lower, upper):
                continue
            ax.fill_betweenx(
                np.r_[0.0, sim.probabilities],
                np.r_[lower[0], lower],
                np.r_[upper[0], upper],
                step="pre",
                color=options.metric_color,
                alpha=options.metric_alpha,
                zorder=0.5,
                label="MAVM supp" if full_support else kind.name,
            )
            _plot_quantile_curve(
                ax,
                lower,
                sim.probabilities,
                color=options.metric_line_color,
                linestyle=options.metric_linestyle,
                linewidth=options.lw,
            )
            _plot_quantile_curve(
                ax,
                upper,
                sim.probabilities,
                color=options.metric_line_color,
                linestyle=options.metric_linestyle,
                linewidth=options.lw,
            )
        elif display is EMetricDisp.INTEGRATION_AREA:
            sim_cdf = calc_ecdf(sim.samples[sim_indices[0]])
            exp_cdf = calc_ecdf(exp.samples[exp_indices[0]])
            if kind is EMetricType.MAVM:
                if not isinstance(metric, MetricMAVM):
                    raise TypeError("MAVM integration shading requires MetricMAVM.")
                mavm = metric.calc(
                    sim.samples[sim_indices[0]], exp.samples[exp_indices[0]]
                )
                exp_bound = (
                    mavm.exp_conf_upper
                    if output is EMetricOut.D_PLUS
                    else mavm.exp_conf_lower
                )
                exp_cdf = ValidECDF(exp_bound, mavm.exp_probs)
                _plot_quantile_curve(
                    ax,
                    mavm.exp_conf_lower,
                    mavm.exp_probs,
                    color=options.exp_color,
                    linestyle=options.exp_conf_linestyle,
                    linewidth=options.lw,
                )
                _plot_quantile_curve(
                    ax,
                    mavm.exp_conf_upper,
                    mavm.exp_probs,
                    color=options.exp_color,
                    linestyle=options.exp_conf_linestyle,
                    linewidth=options.lw,
                )
            probabilities = np.unique(
                np.r_[sim_cdf.probabilities, exp_cdf.probabilities]
            )
            sim_quantiles = _evaluate_ecdf_quantiles(sim_cdf, probabilities)
            exp_quantiles = _evaluate_ecdf_quantiles(exp_cdf, probabilities)
            lower = np.minimum(sim_quantiles, exp_quantiles)
            upper = np.maximum(sim_quantiles, exp_quantiles)
            if kind is EMetricType.MAVM:
                if output is EMetricOut.D_PLUS:
                    lower = sim_quantiles
                else:
                    upper = sim_quantiles
            ax.fill_betweenx(
                np.r_[0.0, probabilities],
                np.r_[lower[0], lower],
                np.r_[upper[0], upper],
                step="pre",
                facecolor=to_rgba(options.metric_color, options.metric_alpha),
                edgecolor=options.metric_line_color,
                linewidth=options.lw,
                label=kind.name,
            )
        elif display is EMetricDisp.CDF_DIFFERENCE:
            if comparison is not None or len(sim_indices) == len(exp_indices) == 1:
                curve_pairs = (
                    (
                        sim.original_quantiles[sim_indices[0]],
                        exp.original_quantiles[exp_indices[0]],
                    ),
                )
            else:
                sim_quantiles = sim.quantiles.reshape(-1, sim.probabilities.size)
                exp_quantiles = exp.quantiles.reshape(-1, exp.probabilities.size)
                curve_pairs = (
                    (
                        np.min(sim_quantiles, axis=0),
                        np.min(exp_quantiles, axis=0),
                    ),
                    (
                        np.max(sim_quantiles, axis=0),
                        np.max(exp_quantiles, axis=0),
                    ),
                )
            for pair_index, (sim_quantiles, exp_quantiles) in enumerate(curve_pairs):
                support = np.unique(np.r_[sim_quantiles, exp_quantiles])
                sim_probabilities = np.r_[0.0, sim.probabilities][
                    np.searchsorted(sim_quantiles, support, side="right")
                ]
                exp_probabilities = np.r_[0.0, exp.probabilities][
                    np.searchsorted(exp_quantiles, support, side="right")
                ]
                index = int(np.argmax(np.abs(sim_probabilities - exp_probabilities)))
                ax.plot(
                    [support[index], support[index]],
                    [sim_probabilities[index], exp_probabilities[index]],
                    color=options.metric_line_color,
                    linestyle=options.metric_linestyle,
                    marker="_",
                    label="KS" if pair_index == 0 else None,
                    linewidth=options.lw,
                )
        else:
            finite = values[np.isfinite(values)]
            if finite.size == 0:
                label = f"{metric_label}: undefined"
            elif finite.size == 1:
                label = f"{metric_label} = {finite[0]:.4g}"
            else:
                label = f"{metric_label} original comparisons: [{np.min(finite):.4g}, {np.max(finite):.4g}]"
            units = (
                plot.valid_case.observables[plot.observable].phys_units
                if metric.get_scale() is EMetricScale.PHYSICAL
                else "-"
            )
            units = "-" if units in ("1", "") else units
            # Place later annotations below earlier ones, preserving composability.
            ax.text(
                0.98,
                0.98 - 0.07 * len(ax.texts),
                f"{label} [{units}]",
                transform=ax.transAxes,
                ha="right",
                va="top",
                fontsize=options.font_leg_size,
            )
        if ax.get_legend_handles_labels()[1]:
            ax.legend()
        _apply_plot_options(plot.figure, ax, options, ecdf=True)


def plot_u_pooling_cdf(
    model_cdfs: list[ValidECDF],
    exp_obs: np.ndarray,
    *,
    valid_plot_opts: ValidPlotOpts | None = None,
    ax: Axes | None = None,
) -> tuple[Figure, Axes]:
    """Show probability-integral transforms against the continuous uniform CDF.

    Each exp observation must have its own corresponding model CDF. This is a
    separate probability-coordinate diagnostic, not a physical-ECDF overlay.
    """
    distance = calc_u_pooling_1d(model_cdfs, exp_obs)
    transformed = []
    for cdf, observation in zip(model_cdfs, exp_obs, strict=True):
        index = np.searchsorted(cdf.quantiles, observation, side="right") - 1
        transformed.append(0.0 if index < 0 else cdf.probabilities[index])
    cdf = calc_ecdf(np.asarray(transformed))
    fig, ax = _create_axes(ax, valid_plot_opts)
    options = ValidPlotOpts() if valid_plot_opts is None else valid_plot_opts
    _plot_quantile_curve(
        ax,
        cdf.quantiles,
        cdf.probabilities,
        color=options.exp_color,
        label="U-pooling",
        linestyle=options.exp_linestyle,
        linewidth=options.lw * options.ecdf_lw_scale,
    )
    ax.plot(
        [0.0, 1.0],
        [0.0, 1.0],
        color=options.sim_color,
        linestyle=options.sim_linestyle,
        label="Uniform reference",
        linewidth=options.lw,
    )
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.0)
    ax.text(
        0.98,
        0.98,
        f"U-pooling = {distance:.4g}",
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=options.font_leg_size,
    )
    ax.set_xlabel("Transformed observation [-]")
    ax.set_ylabel("Cumulative probability")
    ax.legend()
    _apply_plot_options(fig, ax, options, ecdf=True)
    return fig, ax


def plot_valid_metric_time(
    result: ValidResult,
    observable: str,
    metric: EMetricType,
    *,
    selectors: dict[EValidResAxis, int] | None = None,
    output: EMetricOut | None = None,
    valid_plot_opts: ValidPlotOpts | None = None,
    ax: Axes | None = None,
) -> tuple[Figure, Axes]:
    """Plot a selected scenario over time. MAVM defaults to both directions."""
    field = result.metrics[observable][metric]
    outputs = (
        (EMetricOut.D_PLUS, EMetricOut.D_MINUS)
        if metric is EMetricType.MAVM and output is None
        else (field._resolve_output(output),)
    )
    times = result.observables[observable].times
    if times is None:
        raise ValueError("Time plot requires physical time coordinates.")
    traces = [
        extract_metric_values(
            field, keep=EValidResAxis.TIME, selectors=selectors, output=direction
        )
        for direction in outputs
    ]
    fig, ax = _create_axes(ax, valid_plot_opts)
    options = valid_plot_opts or ValidPlotOpts()
    for direction, trace in zip(outputs, traces, strict=True):
        trace_label = (
            _create_metric_label(metric, direction)
            if metric is EMetricType.MAVM
            else metric.name
        )
        negative = direction is EMetricOut.D_MINUS
        ax.plot(
            times,
            trace,
            color=options.metric_line_color,
            marker="s" if negative else "o",
            linestyle=options.metric_alt_linestyle if negative else "-",
            label=trace_label,
        )
    ax.legend()
    time_units = result.observables[observable].time_units
    ax.set_xlabel(f"Time [{time_units}]" if time_units else "Time")
    label = "MAVM" if len(outputs) == 2 else _create_metric_label(metric, outputs[0])
    ax.set_ylabel(f"{label} [{_metric_units(result, observable, metric)}]")
    ax.set_title(observable)
    ax.grid(alpha=0.3)
    _apply_plot_options(fig, ax, options)
    return fig, ax


def plot_metric_dist(
    result: ValidResult,
    observable: str,
    metric: EMetricType,
    *,
    output: EMetricOut | None = None,
    selectors: dict[EValidResAxis, int] | None = None,
    valid_plot_opts: ValidPlotOpts | None = None,
    ax: Axes | None = None,
) -> tuple[Figure, Axes]:
    """Histogram finite metric values over selected physical/scenario cells."""
    field = result.metrics[observable][metric]
    output = field._resolve_output(output)
    selection_idx = _build_metric_selection(field, selectors)
    selected_field = select_metric_output(field, output, selection_idxs=selection_idx)
    values = selected_field.ravel()
    values = values[np.isfinite(values)]
    if not len(values):
        raise ValueError("No finite metric values selected.")
    fig, ax = _create_axes(ax, valid_plot_opts)
    options = valid_plot_opts or ValidPlotOpts()
    ax.hist(
        values,
        bins=min(20, len(values)),
        color=to_rgba(options.metric_color, options.metric_alpha),
        edgecolor=options.metric_line_color,
        linewidth=options.lw,
    )
    ax.set_xlabel(
        f"{_create_metric_label(metric, output)} "
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


def plot_metric_time_points(
    result: ValidResult,
    observable: str,
    metric: EMetricType,
    *,
    point_idxs: tuple[int, ...],
    selectors: dict[EValidResAxis, int] | None = None,
    output: EMetricOut | None = None,
    columns: int = 3,
    valid_plot_opts: ValidPlotOpts | None = None,
) -> tuple[Figure, np.ndarray]:
    """Plot a selected scenario per point. MAVM defaults to both directions."""
    field = result.metrics[observable][metric]
    validation_output = (
        EMetricOut.D_PLUS
        if metric is EMetricType.MAVM and output is None
        else field._resolve_output(output)
    )
    if result.observables[observable].times is None:
        raise ValueError("Time plot requires physical time coordinates.")
    for point_idx in point_idxs:
        selected = dict(selectors or {})
        selected[EValidResAxis.POINT] = point_idx
        extract_metric_values(
            field,
            keep=EValidResAxis.TIME,
            selectors=selected,
            output=validation_output,
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


def plot_metric_extrema_time(
    result: ValidResult,
    observable: str,
    metric: EMetricType,
    *,
    profile: ValidMetricExtrema,
    point_idxs: tuple[int, ...] | None = None,
    selectors: dict[EValidResAxis, int] | None = None,
    output: EMetricOut | None = None,
    columns: int = 3,
    valid_plot_opts: ValidPlotOpts | None = None,
) -> tuple[Figure, np.ndarray]:
    """Plot precomputed minimum and maximum traces in one panel per point.

    Select any remaining non-singleton axes with selectors. By default all
    profiled points and both MAVM directions are shown.
    """
    observable_data = result.observables[observable]
    times = observable_data.times
    if times is None or EValidResAxis.TIME not in profile.axes:
        raise ValueError("Extrema time plots require a retained physical time axis.")
    selected = dict(selectors or {})
    if EValidResAxis.TIME in selected or EValidResAxis.POINT in selected:
        raise ValueError("Use point_idxs to select points and leave time unselected.")

    minimum_field = MetricField(profile.axes, profile.minimum)
    outputs = (
        (EMetricOut.D_PLUS, EMetricOut.D_MINUS)
        if metric is EMetricType.MAVM and output is None
        else (minimum_field._resolve_output(output),)
    )
    point_axis = EValidResAxis.POINT
    if point_axis in profile.axes:
        point_count = profile.minimum[outputs[0]].shape[profile.axes.index(point_axis)]
        available_points = range(point_count)
    else:
        available_points = (profile.selectors[point_axis],)
    points = tuple(available_points) if point_idxs is None else point_idxs
    if not isinstance(points, tuple) or not points:
        raise ValueError("Supply a nonempty tuple of point indices.")

    traces = []
    for point_idx in points:
        if isinstance(point_idx, (bool, np.bool_)) or not isinstance(
            point_idx, (int, np.integer)
        ):
            raise TypeError("Point indices must be integers.")
        if point_idx not in available_points:
            raise IndexError("Point index is outside the extrema profile.")
        if point_axis in profile.axes:
            selected[point_axis] = point_idx
        indices = _build_metric_selection(
            minimum_field, selected, keep=EValidResAxis.TIME
        )
        point_traces = [
            (profile.minimum[direction][indices], profile.maximum[direction][indices])
            for direction in outputs
        ]
        if any(
            lower.shape != times.shape or upper.shape != times.shape
            for lower, upper in point_traces
        ):
            raise ValueError("Extrema traces must match the physical time coordinates.")
        traces.append(point_traces)

    fig, axes = create_valid_plot_grid(
        len(points), columns=columns, valid_plot_opts=valid_plot_opts
    )
    options = valid_plot_opts or ValidPlotOpts()
    time_units = observable_data.time_units
    metric_label = (
        "MAVM" if len(outputs) == 2 else _create_metric_label(metric, outputs[0])
    )
    for point_idx, point_traces, ax in zip(points, traces, axes.flat, strict=False):
        for direction, (lower, upper) in zip(outputs, point_traces, strict=True):
            label = (
                _create_metric_label(metric, direction)
                if metric is EMetricType.MAVM
                else metric.name
            )
            marker = "s" if direction is EMetricOut.D_MINUS else "o"
            ax.plot(
                times,
                lower,
                marker=marker,
                color=options.metric_line_color,
                label=f"{label} min",
            )
            ax.plot(
                times,
                upper,
                marker=marker,
                linestyle=options.metric_alt_linestyle,
                color=options.metric_line_color,
                label=f"{label} max",
            )
        ax.set_xlabel(f"Time [{time_units}]" if time_units else "Time")
        ax.set_ylabel(f"{metric_label} [{_metric_units(result, observable, metric)}]")
        ax.set_title(f"Point {point_idx}: {observable}")
        ax.grid(alpha=0.3)
        ax.legend()
        _apply_plot_options(fig, ax, options)
    return fig, axes


def plot_metric_extrema_ecdfs(
    valid_case: ValidCase,
    observable: str,
    metric: EMetricType,
    *,
    profile: ValidMetricExtrema,
    selectors: dict[EValidResAxis, int] | None = None,
    output: EMetricOut | None = None,
    valid_plot_opts: ValidPlotOpts | None = None,
) -> tuple[Figure, np.ndarray]:
    """Plot the original ECDF pairs that produced a stored minimum and maximum.

    Select every non-singleton retained axis, including the physical time if
    present. Multi-output profiles require an explicit metric output.
    """
    field = MetricField(profile.axes, profile.minimum)
    output = field._resolve_output(output)
    values = profile.minimum[output]
    missing_selectors = set(selectors or {}) - set(profile.axes)
    if missing_selectors:
        raise ValueError("Selectors must name retained profile axes.")
    selected = dict(selectors or {})
    for axis, size in zip(profile.axes, values.shape, strict=True):
        if size == 1:
            selected.setdefault(axis, 0)
        elif axis not in selected:
            raise ValueError(f"Select a {axis.value} index for the extrema ECDFs.")
    indices = _build_metric_selection(field, selected)
    cell = {**profile.selectors, **selected}
    panels = []
    for bound, scores, origins in (
        ("Minimum", values, profile.minimum_indices[output]),
        ("Maximum", profile.maximum[output], profile.maximum_indices[output]),
    ):
        score = float(scores[indices])
        origin = origins[indices]
        if not np.isfinite(score) or np.any(origin < 0):
            raise ValueError(f"The selected {bound.lower()} has no finite comparison.")
        comparison_indices = {
            **cell,
            **dict(zip(profile.reduced_axes, map(int, origin), strict=True)),
        }
        panels.append((bound, score, comparison_indices))

    fig, axes = create_valid_plot_grid(2, rows=1, valid_plot_opts=valid_plot_opts)
    for ax, (bound, score, comparison_indices) in zip(axes.flat, panels, strict=True):
        sim_trans = comparison_indices.get(EValidResAxis.SIM_TRANS, 0)
        sim_nuis = comparison_indices.get(EValidResAxis.SIM_NUIS, 0)
        exp_trans = comparison_indices.get(EValidResAxis.EXP_TRANS, 0)
        exp_nuis = comparison_indices.get(EValidResAxis.EXP_NUIS, 0)
        sim_offset = EEpisOffset(comparison_indices.get(EValidResAxis.SIM_OFFSET, 0))
        exp_offset = EEpisOffset(comparison_indices.get(EValidResAxis.EXP_OFFSET, 0))
        plot_sim_exp_cdfs(
            valid_case,
            observable,
            sim_view=ValidECDFView(trans_idxs=(sim_trans,), nuis_idxs=(sim_nuis,)),
            exp_view=ValidECDFView(trans_idxs=(exp_trans,), nuis_idxs=(exp_nuis,)),
            sim_offset=sim_offset,
            exp_offset=exp_offset,
            point_idxs=(comparison_indices.get(EValidResAxis.POINT, 0),),
            component_idx=comparison_indices.get(EValidResAxis.COMPONENT, 0),
            time_idx=comparison_indices.get(EValidResAxis.TIME, 0),
            valid_plot_opts=valid_plot_opts,
            ax=ax,
        )
        sim_case = f"Sim. trans. {sim_trans}, nuis. {sim_nuis}"
        exp_case = f"Exp. trans. {exp_trans}, nuis. {exp_nuis}"
        if EValidResAxis.SIM_OFFSET in comparison_indices:
            sim_case += f", {sim_offset.name.lower()} offset"
        if EValidResAxis.EXP_OFFSET in comparison_indices:
            exp_case += f", {exp_offset.name.lower()} offset"
        label = (
            _create_metric_label(metric, output)
            if metric is EMetricType.MAVM
            else metric.name
        )
        ax.set_title(f"{bound} {label} = {score:.4g}\n{sim_case}\n{exp_case}")
    return fig, axes


def _plot_quantile_curve(
    ax: Axes,
    quantiles: np.ndarray,
    probabilities: np.ndarray,
    *,
    color: str,
    label: str | None = None,
    linestyle: _LineStyle,
    linewidth: float,
    alpha: float = 1.0,
) -> None:
    ax.step(
        np.r_[quantiles[0], quantiles],
        np.r_[0.0, probabilities],
        where="post",
        color=color,
        label=label,
        linestyle=linestyle,
        linewidth=linewidth,
        alpha=alpha,
    )


def _plot_distribution_summary(
    ax: Axes,
    summary: ValidECDFSummary,
    *,
    color: str,
    label: str,
    options: ValidPlotOpts,
    bounds_linestyle: _LineStyle,
    linestyle: _LineStyle,
) -> None:
    _plot_quantile_curve(
        ax,
        summary.nominal,
        summary.probabilities,
        color=color,
        label=label,
        linestyle=linestyle,
        linewidth=options.lw * options.ecdf_lw_scale,
    )
    _plot_quantile_curve(
        ax,
        summary.lower,
        summary.probabilities,
        color=color,
        linestyle=bounds_linestyle,
        linewidth=options.lw * options.ecdf_lw_scale,
    )
    _plot_quantile_curve(
        ax,
        summary.upper,
        summary.probabilities,
        color=color,
        linestyle=bounds_linestyle,
        linewidth=options.lw * options.ecdf_lw_scale,
    )
    if options.shade_bounds and not np.array_equal(summary.lower, summary.upper):
        ax.fill_betweenx(
            np.r_[0.0, summary.probabilities],
            np.r_[summary.lower[0], summary.lower],
            np.r_[summary.upper[0], summary.upper],
            step="pre",
            color=color,
            alpha=options.bounds_alpha,
            zorder=0.4,
        )


def plot_valid_mavm_diagnostic(
    sim: ValidECDFSummary,
    exp: ValidECDFSummary | None = None,
    *,
    correction: MAVMBoundPairs | None = None,
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
    if exp is not None and view is not EValidMAVMDiagnostic.MODEL_CORRECTION:
        _plot_distribution_summary(
            ax,
            exp,
            color=options.exp_color,
            label=options.exp_label,
            options=options,
            linestyle=options.exp_linestyle,
            bounds_linestyle=options.exp_bounds_linestyle,
        )
    _plot_distribution_summary(
        ax,
        sim,
        color=options.sim_color,
        label=options.sim_label,
        options=options,
        linestyle=options.sim_linestyle,
        bounds_linestyle=options.sim_bounds_linestyle,
    )
    if correction is not None and view is EValidMAVMDiagnostic.COMPARISONS:
        for comparison, linestyle, label in (
            (correction.upper, options.metric_linestyle, "Upper case"),
            (correction.lower, options.metric_alt_linestyle, "Lower case"),
        ):
            _plot_quantile_curve(
                ax,
                comparison.model_quantiles + comparison.d_plus,
                comparison.model_probs,
                color=options.metric_line_color,
                linestyle=linestyle,
                label=label,
                linewidth=options.lw,
            )
            _plot_quantile_curve(
                ax,
                comparison.model_quantiles - comparison.d_minus,
                comparison.model_probs,
                color=options.metric_line_color,
                linestyle=linestyle,
                linewidth=options.lw,
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
                color=options.metric_color,
                alpha=options.metric_alpha,
                zorder=0.5,
                label="MAVM supp",
            )
        _plot_quantile_curve(
            ax,
            lower,
            probabilities,
            color=options.metric_line_color,
            linestyle=options.metric_linestyle,
            label=None if options.shade_bounds else "MAVM supp",
            linewidth=options.lw,
        )
        _plot_quantile_curve(
            ax,
            upper,
            probabilities,
            color=options.metric_line_color,
            linestyle=options.metric_linestyle,
            linewidth=options.lw,
        )
    ax.set_ylabel("Cumulative probability")
    ax.set_ylim(0.0, 1.02)
    if ax.get_legend_handles_labels()[1]:
        ax.legend()
    _apply_plot_options(fig, ax, valid_plot_opts, ecdf=True)
    return fig, ax
