# ==============================================================================
# pyvale: the python validation engine
# License: MIT
# Copyright (C) 2026 The Computer Aided Validation Team
# ==============================================================================

"""Diagnostic plots for validation samples and labelled metric fields."""

from dataclasses import dataclass, field, replace
from enum import Enum, StrEnum, auto
from math import ceil, sqrt

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.colors import to_rgba
from matplotlib.figure import Figure
from matplotlib.ticker import MaxNLocator, ScalarFormatter

from pyvale.sensorsim.visualopts import PlotOptsGeneral
from pyvale.validation.ecdftypes import (
    ECDFMetricBounds,
    EECDFReduce,
    ECDF,
    ECDFViewSel,
    ECDFView,
)
from pyvale.validation.metricstrategy import MetricMAVM
from pyvale.validation.imetric import (
    EMetricOut,
    EMetricScale,
    EMetricType,
    IValidMetric,
    extract_metric_outputs,
)
from pyvale.validation.ecdfops import (
    _evaluate_ecdf_quantiles,
    _reduce_ecdf_quantiles,
    calc_ecdf,
    calc_ecdf_epis_bounds,
    calc_ecdf_view,
)
from pyvale.validation.validprof import _mavm_confidence_shifts
from pyvale.validation.validtools import (
    _build_metric_selection,
    _create_default_metric_labels,
    _create_metric_label,
    extract_metric_values,
    select_metric_output,
)
from pyvale.validation.validtypes import (
    EEpisOffset,
    EValidAxis,
    EValidResAxis,
    MAVMBoundPairs,
    MetricField,
    ValidCase,
    ValidMetricExtrema,
    ValidResult,
    _get_axis_size,
)

_LineStyle = str | tuple[float, tuple[float, ...]]


@dataclass(slots=True)
class ValidPlotOpts:
    """Control validation figure layout, labels, source styles and ECDF shading.

    Attributes
    ----------
    title : str or None
        Optional axes title override.
    x_label : str or None
        Optional x-axis label override.
    x_axis_offset : bool
        Allow an additive offset in numeric x-axis tick labels.
    y_label : str or None
        Optional y-axis label override.
    sim_label : str
        Base legend label for simulation curves.
    exp_label : str
        Base legend label for experimental curves.
    metric_labels : dict[EMetricType, str]
        Display label overrides keyed by metric type.
    sim_color : str
        Colour of simulation curves.
    exp_color : str
        Colour of experimental curves.
    sim_profile_color : str
        Colour of simulation nuisance profile curves.
    exp_profile_color : str
        Colour of experimental nuisance profile curves.
    sim_linestyle : _LineStyle
        Line style of nominal simulation curves.
    exp_linestyle : _LineStyle
        Line style of nominal experimental curves.
    sim_bounds_linestyle : _LineStyle
        Line style of simulation bounds.
    exp_bounds_linestyle : _LineStyle
        Line style of experimental bounds.
    sim_profile_linestyle : _LineStyle
        Line style of simulation profile bounds.
    exp_profile_linestyle : _LineStyle
        Line style of experimental profile bounds.
    sim_offset_linestyle : _LineStyle
        Line style of simulation offset curves.
    exp_offset_linestyle : _LineStyle
        Line style of experimental offset curves.
    exp_conf_linestyle : _LineStyle
        Line style of experimental confidence limits.
    shade_bounds : bool
        Fill between distinct lower and upper bounds.
    bounds_alpha : float
        Opacity of the region between ECDF bounds.
    metric_color : str
        Fill colour for metric overlays and histograms.
    metric_line_color : str
        Line and edge colour for metric artists.
    metric_alpha : float
        Opacity of shaded metric regions.
    metric_linestyle : _LineStyle
        Line style for the primary metric curve.
    metric_alt_linestyle : _LineStyle
        Line style for a second metric curve or direction.
    ecdf_overlay_alpha : float
        Opacity of individual ECDF, profile and confidence overlays.
    legend_outside : bool
        Place the legend below the axes.
    legend_columns : int
        Number of legend columns.
    resolution : float
        Figure resolution in dots per inch.
    font_tick_size : float
        Tick label size in points.
    font_head_size : float
        Axes title size in points.
    font_ax_size : float
        Axis label size in points.
    font_leg_size : float
        Default legend font size in points.
    font_ecdf_leg_size : float
        ECDF legend font size in points.
    font_def_weight : str
        Font weight for titles and axis labels.
    font_name : str
        Matplotlib font family.
    single_fig_size_landscape : tuple[float, float]
        Width and height of one landscape panel, in inches.
    lw : float
        Base line width in points.
    ecdf_lw_scale : float
        Multiplier applied to the base ECDF line width.
    ms : float
        Marker size in points.
    """

    title: str | None = None
    x_label: str | None = None
    x_axis_offset: bool = True
    y_label: str | None = None
    sim_label: str = "Sim"
    exp_label: str = "Exp"
    metric_labels: dict[EMetricType, str] = field(
        default_factory=_create_default_metric_labels
    )

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
    ecdf_overlay_alpha: float = 0.25

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


class EValidSource(StrEnum):
    """Choose simulation or experiment as the source for an ECDF plot."""

    SIM = "sim"
    EXP = "exp"


class EValidMAVMDiagnostic(Enum):
    """Choose distribution, pairwise comparison or support correction views."""

    DISTRIBUTIONS = auto()
    COMPARISONS = auto()
    CORRECTION = auto()
    MODEL_CORRECTION = auto()


class EECDFCentre(Enum):
    """Choose whether to draw the mean or median quantile curve across cases."""

    NONE = auto()
    MEAN = auto()
    MEDIAN = auto()


@dataclass(slots=True)
class ECDFVisOpts:
    """Choose which curves to draw from an ECDF view.

    Overlay indices are original (transferable, nuisance) pairs.
    An empty tuple draws no overlays. None draws all selected cases.

    Attributes
    ----------
    show_bounds : bool
        Draw pointwise lower and upper quantiles for a multi-curve view.
    centre : EECDFCentre
        Optional pointwise mean or median of the displayed quantile curves.
    epis_case_idxs : tuple[tuple[int, int], ...] or None
        Original (transferable, nuisance) index pairs to overlay. None selects every
        pair; an empty tuple selects none.
    """

    show_bounds: bool = True
    centre: EECDFCentre = EECDFCentre.NONE
    epis_case_idxs: tuple[tuple[int, int], ...] | None = ()


class EShowECDFBounds(Enum):
    """Choose outer bounds or all individual cases for ECDF display."""

    OUTER = auto()
    ALL = auto()


class EMetricDisp(Enum):
    """Choose how a validation metric is displayed on an ECDF plot."""

    SUPPORT_SHIFT = auto()
    INTEGRATION_AREA = auto()
    CDF_DIFFERENCE = auto()
    ANNOTATION = auto()


@dataclass(slots=True)
class ECDFReference:
    """Normalise by a quantile of the displayed family's mean quantile curve.

    Attributes
    ----------
    source : EValidSource
        Source used to obtain the normalising quantile.
    probability : float
        Cumulative probability in (0, 1] on the selected mean quantile curve.
    """

    source: EValidSource = EValidSource.SIM
    probability: float = 0.5


@dataclass(slots=True)
class ECDFComparison:
    """Choose an original conditional sim/exp pair for a metric overlay.

    Attributes
    ----------
    sim_trans_idx : int
        Original simulation transferable case index.
    sim_nuis_idx : int
        Original simulation nuisance case index.
    exp_trans_idx : int
        Original experimental transferable case index.
    exp_nuis_idx : int
        Original experimental nuisance case index.
    sim_offset : EEpisOffset
        Simulation offset used in the comparison.
    exp_offset : EEpisOffset
        Experimental offset used in the comparison.
    """

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

    Attributes
    ----------
    figure : Figure
        Matplotlib figure for the composed plot.
    axes : ndarray
        Object array of Matplotlib axes with shape (n_rows, n_columns). Active
        panels follow point_idxs in row-major order.
    valid_case : ValidCase
        Aligned simulation and experiment observations with their named axes.
    observable : str
        Observable tag identifying matching metadata and observations.
    point_idxs : tuple[int, ...]
        Original point indices in panel order. None selects all available points.
    component_idx : int
        Zero-based observable component index.
    time_idx : int
        Zero-based evaluation time index.
    data : tuple[dict[EValidSource, ECDFView], ...]
        One dictionary per active panel, mapping each plotted source to its computed
        ECDF view.
    options : ValidPlotOpts
        Matplotlib styling options used by the plot.
    normalisation : tuple[float, ...] or None
        Optional nonzero normalisation constant for each active panel.
    """

    figure: Figure
    axes: np.ndarray
    valid_case: ValidCase
    observable: str
    point_idxs: tuple[int, ...]
    component_idx: int
    time_idx: int
    data: tuple[dict[EValidSource, ECDFView], ...]
    options: ValidPlotOpts
    normalisation: tuple[float, ...] | None = None


def _apply_plot_options(
    fig: Figure,
    ax: Axes,
    valid_plot_opts: ValidPlotOpts | None,
    *,
    ecdf: bool = False,
) -> None:
    """Apply figure resolution, formatting and legend settings to an axes.

    Parameters
    ----------
    fig : Figure
        Figure containing the axes.
    ax : Axes
        Matplotlib axes to draw into. Public plotters create axes when None is
        supplied.
    valid_plot_opts : ValidPlotOpts or None
        Matplotlib layout and appearance settings. Use None for defaults.
    ecdf : bool, optional
        Use ECDF-specific legend sizes and preserve the width of existing ECDF
        lines.
    """
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
    ax.xaxis.set_major_formatter(ScalarFormatter(useOffset=options.x_axis_offset))
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
    """Reuse existing axes or create a single diagnostic panel.

    Parameters
    ----------
    ax : Axes or None
        Matplotlib axes to draw into. Public plotters create axes when None is
        supplied.
    valid_plot_opts : ValidPlotOpts or None
        Matplotlib layout and appearance settings. Use None for defaults.

    Returns
    -------
    tuple[Figure, Axes]
        Figure and axes for the requested diagnostic.
    """
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
    """Resolve the units displayed for a metric.

    Parameters
    ----------
    result : ValidResult
        Validation results containing the requested observable and metric.
    observable : str
        Observable tag identifying matching metadata and observations.
    metric : EMetricType
        Metric identifier or strategy used by the plot or overlay.

    Returns
    -------
    str
        Observable units for a physical metric, or '-' for a dimensionless or
        unspecified unit.
    """
    if result.metric_configs[metric].get_scale() is not EMetricScale.PHYSICAL:
        return "-"

    units = result.observables[observable].phys_units
    return "-" if units in ("1", "") else units


def plot_sim_ecdfs(
    valid_case: ValidCase,
    observable: str,
    *,
    view: ECDFViewSel | None = None,
    vis_opts: ECDFVisOpts | None = None,
    offset: EEpisOffset | None = EEpisOffset.NOMINAL,
    show_bounds: EShowECDFBounds = EShowECDFBounds.OUTER,
    nuisance_profile: EECDFReduce | None = None,
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
    within the selected envelope. ALL also displays original epistemic curves.

    Parameters
    ----------
    valid_case : ValidCase
        Aligned simulation and experiment observations with their named axes.
    observable : str
        Observable tag identifying matching metadata and observations.
    view : ECDFViewSel or None, optional
        Epistemic selection and reduction for this source. None uses the default
        selection.
    vis_opts : ECDFVisOpts or None, optional
        Controls for drawing bounds, a centre curve and original cases.
    offset : EEpisOffset or None, optional
        Offset to plot. None includes nominal and all available coherent offsets.
    show_bounds : EShowECDFBounds, optional
        Display mode for outer limits or individual offset and case curves.
    nuisance_profile : EECDFReduce or None, optional
        Optional nuisance reduction drawn as an additional curve or set of curves.
    nuisance_profile_quantile : float or None, optional
        Probability in [0, 1] when nuisance_profile is QUANTILE.
    point_idxs : tuple[int, ...] or None, optional
        Original point indices in panel order. None selects all available points.
    component_idx : int, optional
        Zero-based observable component index.
    time_idx : int, optional
        Zero-based evaluation time index.
    rows : int or None, optional
        Requested grid rows. None chooses the count from the panel count and
        columns.
    columns : int or None, optional
        Requested grid columns. None chooses the count automatically.
    valid_plot_opts : ValidPlotOpts or None, optional
        Matplotlib layout and appearance settings. Use None for defaults.
    ax : Axes or None, optional
        Matplotlib axes to draw into. Public plotters create axes when None is
        supplied.

    Returns
    -------
    ECDFPlot
        Figure, two-dimensional axes grid and the simulation ECDF view for each
        active panel.
    """
    return _plot_ecdf_sources(
        valid_case,
        observable,
        ((EValidSource.SIM, view, vis_opts, offset),),
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
    view: ECDFViewSel | None = None,
    vis_opts: ECDFVisOpts | None = None,
    offset: EEpisOffset | None = EEpisOffset.NOMINAL,
    show_bounds: EShowECDFBounds = EShowECDFBounds.OUTER,
    confidence_alpha: float | None = None,
    nuisance_profile: EECDFReduce | None = None,
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
    the enclosing bounds, while ALL includes each selected original epistemic curve.

    Parameters
    ----------
    valid_case : ValidCase
        Aligned simulation and experiment observations with their named axes.
    observable : str
        Observable tag identifying matching metadata and observations.
    view : ECDFViewSel or None, optional
        Epistemic selection and reduction for this source. None uses the default
        selection.
    vis_opts : ECDFVisOpts or None, optional
        Controls for drawing bounds, a centre curve and original cases.
    offset : EEpisOffset or None, optional
        Offset to plot. None includes nominal and all available coherent offsets.
    show_bounds : EShowECDFBounds, optional
        Display mode for outer limits or individual offset and case curves.
    confidence_alpha : float or None, optional
        Experimental MAVM confidence parameter in (0, 0.5]. None suppresses
        confidence bounds.
    nuisance_profile : EECDFReduce or None, optional
        Optional nuisance reduction drawn as an additional curve or set of curves.
    nuisance_profile_quantile : float or None, optional
        Probability in [0, 1] when nuisance_profile is QUANTILE.
    point_idxs : tuple[int, ...] or None, optional
        Original point indices in panel order. None selects all available points.
    component_idx : int, optional
        Zero-based observable component index.
    time_idx : int, optional
        Zero-based evaluation time index.
    rows : int or None, optional
        Requested grid rows. None chooses the count from the panel count and
        columns.
    columns : int or None, optional
        Requested grid columns. None chooses the count automatically.
    valid_plot_opts : ValidPlotOpts or None, optional
        Matplotlib layout and appearance settings. Use None for defaults.
    ax : Axes or None, optional
        Matplotlib axes to draw into. Public plotters create axes when None is
        supplied.

    Returns
    -------
    ECDFPlot
        Figure, two-dimensional axes grid and the experimental ECDF view for each
        active panel.
    """
    return _plot_ecdf_sources(
        valid_case,
        observable,
        ((EValidSource.EXP, view, vis_opts, offset),),
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
    sim_view: ECDFViewSel | None = None,
    exp_view: ECDFViewSel | None = None,
    sim_vis_opts: ECDFVisOpts | None = None,
    exp_vis_opts: ECDFVisOpts | None = None,
    sim_offset: EEpisOffset | None = EEpisOffset.NOMINAL,
    exp_offset: EEpisOffset | None = EEpisOffset.NOMINAL,
    show_bounds: EShowECDFBounds = EShowECDFBounds.OUTER,
    confidence_alpha: float | None = None,
    nuisance_profile: EECDFReduce | None = None,
    nuisance_profile_quantile: float | None = None,
    point_idxs: tuple[int, ...] | None = (0,),
    component_idx: int = 0,
    time_idx: int = 0,
    rows: int | None = None,
    columns: int | None = None,
    valid_plot_opts: ValidPlotOpts | None = None,
    ax: Axes | None = None,
) -> ECDFPlot:
    """Compare sim and exp ECDFs across the selected epistemic cases.

    Each source defaults to a shaded pointwise epistemic envelope, or one ECDF
    when only one epistemic curve remains. Set nuisance_profile to overlay a pointwise
    nuisance reduction within each transferable case. QUANTILE also requires
    nuisance_profile_quantile. Use exp_offset=None or sim_offset=None to display
    nominal and coherent offsets together. Set confidence_alpha to display
    MAVM experimental confidence bounds. ALL includes individual epistemic curves
    and confidence pairs. Metric overlays use the original plot selections.

    Parameters
    ----------
    valid_case : ValidCase
        Aligned simulation and experiment observations with their named axes.
    observable : str
        Observable tag identifying matching metadata and observations.
    sim_view : ECDFViewSel or None, optional
        Simulation epistemic selection and reduction settings.
    exp_view : ECDFViewSel or None, optional
        Experimental epistemic selection and reduction settings.
    sim_vis_opts : ECDFVisOpts or None, optional
        Display options for the simulation curves.
    exp_vis_opts : ECDFVisOpts or None, optional
        Display options for the experimental curves.
    sim_offset : EEpisOffset or None, optional
        Simulation offset to display. None includes all available offsets.
    exp_offset : EEpisOffset or None, optional
        Experimental offset to display. None includes all available offsets.
    show_bounds : EShowECDFBounds, optional
        Display mode for outer limits or individual offset and case curves.
    confidence_alpha : float or None, optional
        Experimental MAVM confidence parameter in (0, 0.5]. None suppresses
        confidence bounds.
    nuisance_profile : EECDFReduce or None, optional
        Optional nuisance reduction drawn as an additional curve or set of curves.
    nuisance_profile_quantile : float or None, optional
        Probability in [0, 1] when nuisance_profile is QUANTILE.
    point_idxs : tuple[int, ...] or None, optional
        Original point indices in panel order. None selects all available points.
    component_idx : int, optional
        Zero-based observable component index.
    time_idx : int, optional
        Zero-based evaluation time index.
    rows : int or None, optional
        Requested grid rows. None chooses the count from the panel count and
        columns.
    columns : int or None, optional
        Requested grid columns. None chooses the count automatically.
    valid_plot_opts : ValidPlotOpts or None, optional
        Matplotlib layout and appearance settings. Use None for defaults.
    ax : Axes or None, optional
        Matplotlib axes to draw into. Public plotters create axes when None is
        supplied.

    Returns
    -------
    ECDFPlot
        Figure and axes grid of shape (n_rows, n_columns), with both source views
        per active panel.
    """
    return _plot_ecdf_sources(
        valid_case,
        observable,
        (
            (EValidSource.EXP, exp_view, exp_vis_opts, exp_offset),
            (EValidSource.SIM, sim_view, sim_vis_opts, sim_offset),
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
    sources: tuple[
        tuple[
            EValidSource, ECDFViewSel | None, ECDFVisOpts | None, EEpisOffset | None
        ],
        ...,
    ],
    *,
    show_bounds: EShowECDFBounds,
    confidence_alpha: float | None,
    nuisance_profile: EECDFReduce | None,
    nuisance_profile_quantile: float | None,
    point_idxs: tuple[int, ...] | None,
    component_idx: int,
    time_idx: int,
    rows: int | None,
    columns: int | None,
    valid_plot_opts: ValidPlotOpts | None,
    ax: Axes | None,
) -> ECDFPlot:
    """Prepare the selected views and render one panel per physical point.

    Parameters
    ----------
    valid_case : ValidCase
        Aligned simulation and experiment observations with their named axes.
    observable : str
        Observable tag identifying matching metadata and observations.
    sources : tuple[tuple[EValidSource, ECDFViewSel or None, ECDFVisOpts or None, EEpisOffset or None], ...]
        Tuples of source, computational selection, display options and offset.
    show_bounds : EShowECDFBounds
        Display mode for outer limits or individual offset and case curves.
    confidence_alpha : float or None
        Experimental MAVM confidence parameter in (0, 0.5]. None suppresses
        confidence bounds.
    nuisance_profile : EECDFReduce or None
        Optional nuisance reduction drawn as an additional curve or set of curves.
    nuisance_profile_quantile : float or None
        Probability in [0, 1] when nuisance_profile is QUANTILE.
    point_idxs : tuple[int, ...] or None
        Original point indices in panel order. None selects all available points.
    component_idx : int
        Zero-based observable component index.
    time_idx : int
        Zero-based evaluation time index.
    rows : int or None
        Requested grid rows. None chooses the count from the panel count and
        columns.
    columns : int or None
        Requested grid columns. None chooses the count automatically.
    valid_plot_opts : ValidPlotOpts or None
        Matplotlib layout and appearance settings. Use None for defaults.
    ax : Axes or None
        Matplotlib axes to draw into. Public plotters create axes when None is
        supplied.

    Returns
    -------
    ECDFPlot
        Composed figure and the computed source views for each panel.
    """
    if not isinstance(show_bounds, EShowECDFBounds):
        raise TypeError("show_bounds must be EShowECDFBounds.")

    if confidence_alpha is not None and (
        isinstance(confidence_alpha, (bool, np.bool_))
        or not isinstance(confidence_alpha, (int, float, np.integer, np.floating))
        or not 0 < confidence_alpha <= 0.5
    ):
        raise ValueError("Confidence alpha must be in (0, 0.5].")

    if nuisance_profile is not None:
        if not isinstance(nuisance_profile, EECDFReduce):
            raise TypeError("Nuisance profile must be EECDFReduce.")

        if nuisance_profile is EECDFReduce.KEEP:
            raise ValueError("Nuisance profile requires mean, median or quantile.")

    if nuisance_profile is EECDFReduce.QUANTILE:
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
    point_count = _get_axis_size(valid_case.sim[observable], EValidAxis.POINT)
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
    prepared: list[
        list[tuple[EValidSource, ECDFView, ECDFVisOpts, tuple[int, ...]]]
    ] = []
    profiles: list[list[tuple[EValidSource, ECDFView]]] = []

    for point in selected_points:
        point_sources = []
        point_profiles = []

        for source, view, vis_opts, offset in sources:
            selected_vis = ECDFVisOpts() if vis_opts is None else vis_opts

            if not isinstance(selected_vis, ECDFVisOpts):
                raise TypeError("ECDF visualisation options must be ECDFVisOpts.")

            if show_bounds is EShowECDFBounds.ALL:
                selected_vis = replace(selected_vis, epis_case_idxs=None)

            quantity = (
                valid_case.sim if source is EValidSource.SIM else valid_case.exp
            )[observable]
            curves = calc_ecdf_view(
                quantity,
                view=view,
                point_idx=point,
                component_idx=component_idx,
                time_idx=time_idx,
                offset=offset,
            )
            overlay_indices = _select_ecdf_overlays(curves, selected_vis)

            if (
                np.prod(curves.quants.shape[:-1]) > 1
                and not selected_vis.show_bounds
                and selected_vis.centre is EECDFCentre.NONE
                and not overlay_indices
            ):
                raise ValueError("Select bounds, a centre curve or case overlays.")

            point_sources.append((source, curves, selected_vis, overlay_indices))
            selected_view = curves.opts

            if nuisance_profile is not None:
                if (
                    not selected_vis.show_bounds
                    or selected_view.nuis_reduce is not EECDFReduce.KEEP
                    or selected_view.trans_reduce is not EECDFReduce.KEEP
                ):
                    raise ValueError("Nuisance profiles require an unreduced envelope.")

                if curves.quants.shape[1] == 1:
                    continue

                profile_view = replace(
                    selected_view,
                    nuis_reduce=nuisance_profile,
                    nuis_quantile=nuisance_profile_quantile,
                )
                profile_quantiles = _reduce_ecdf_quantiles(
                    curves.quants, profile_view
                )
                profile_curves = replace(
                    curves,
                    quants=profile_quantiles,
                    opts=profile_view,
                )
                point_profiles.append((source, profile_curves))

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
        profile_by_source = {source: curves for source, curves in point_profiles}

        for source, curves, vis_opts, overlay_indices in point_sources:
            has_offsets = curves.quants.shape[2] > 1
            _plot_ecdf_view(
                panel,
                source,
                curves,
                options,
                vis_opts=vis_opts,
                overlay_indices=overlay_indices,
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

        for source, curves in point_profiles:
            if curves.quants.shape[2] > 1 and show_bounds is EShowECDFBounds.OUTER:
                curves = replace(curves, quants=curves.quants[:, :, :1])

            _plot_ecdf_view(
                panel,
                source,
                curves,
                options,
                vis_opts=ECDFVisOpts(),
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
            {source: curves for source, curves, _, _ in point_sources}
            for point_sources in prepared
        ),
        options,
    )


def _ecdf_profile_label(reduction: EECDFReduce, quantile: float | None) -> str:
    """Format the label suffix for an epistemic reduction.

    Parameters
    ----------
    reduction : EECDFReduce
        Epistemic reduction applied to the quantile curves.
    quantile : float or None
        Quantile level for QUANTILE reduction, if used.

    Returns
    -------
    str
        Short label suffix, or an empty string for an unreduced axis.
    """
    if reduction is EECDFReduce.MEDIAN:
        return "Prof-Med"

    if reduction is EECDFReduce.MEAN:
        return "Prof-Mean"

    if reduction is EECDFReduce.QUANTILE:
        assert quantile is not None
        return f"Prof-Q{100 * quantile:g}"

    return ""


def _select_ecdf_overlays(curves: ECDFView, vis_opts: ECDFVisOpts) -> tuple[int, ...]:
    """Validate selected case pairs and locate their original curves.

    Parameters
    ----------
    curves : ECDFView
        Aligned ECDF quantiles and original case metadata for a single physical
        cell.
    vis_opts : ECDFVisOpts
        Controls for drawing bounds, a centre curve and original cases.

    Returns
    -------
    tuple[int, ...]
        Flat row indices into orig_quants, in requested pair and offset order.
    """
    if not isinstance(vis_opts.centre, EECDFCentre):
        raise TypeError("ECDF centre must be EECDFCentre.")

    pairs = vis_opts.epis_case_idxs

    if pairs is None:
        return tuple(range(len(curves.orig_epis_offsets)))

    if not isinstance(pairs, tuple):
        raise TypeError("Epistemic case selections must be a tuple of index pairs.")

    if not pairs:
        return ()

    case_rows: dict[tuple[int, int], list[int]] = {}

    for row, (trans_idx, nuis_idx, _) in enumerate(curves.orig_epis_offsets):
        case_rows.setdefault((trans_idx, nuis_idx), []).append(row)

    selected_rows: list[int] = []
    seen: set[tuple[int, int]] = set()

    for pair in pairs:
        if not isinstance(pair, tuple) or len(pair) != 2:
            raise ValueError("Select cases using (transferable, nuisance) index pairs.")

        if any(
            isinstance(index, (bool, np.bool_))
            or not isinstance(index, (int, np.integer))
            for index in pair
        ):
            raise TypeError("Epistemic case indices must be integers.")

        if pair not in case_rows:
            raise IndexError("Case index pair is outside the current selection.")

        if pair in seen:
            raise ValueError("Epistemic case index pairs must be unique.")

        seen.add(pair)
        selected_rows.extend(case_rows[pair])

    return tuple(selected_rows)


def _plot_ecdf_view(
    ax: Axes,
    source: EValidSource,
    curves: ECDFView,
    options: ValidPlotOpts,
    *,
    vis_opts: ECDFVisOpts,
    overlay_indices: tuple[int, ...] = (),
    label_suffix: str = "",
    profile: bool = False,
    offset_bounds: bool = False,
    show_all: bool = False,
) -> None:
    """Render the reduced curves and selected original overlays for one source.

    Parameters
    ----------
    ax : Axes
        Matplotlib axes to draw into. Public plotters create axes when None is
        supplied.
    source : EValidSource
        Simulation or experiment source, used to select labels and styles.
    curves : ECDFView
        Aligned ECDF quantiles and original case metadata for a single physical
        cell.
    options : ValidPlotOpts
        Matplotlib styling options used by the plot.
    vis_opts : ECDFVisOpts
        Controls for drawing bounds, a centre curve and original cases.
    overlay_indices : tuple[int, ...], optional
        Flattened original curve indices in display order.
    label_suffix : str, optional
        Additional text appended to the source legend label.
    profile : bool, optional
        Apply the styling for nuisance profile curves.
    offset_bounds : bool, optional
        Apply offset-specific line styling to the bounds.
    show_all : bool, optional
        Draw the individual profile curves as well as their limits.
    """
    view = curves.opts
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

    quantiles = curves.quants.reshape(-1, curves.probs.size)
    single = quantiles.shape[0] == 1
    reduced = (
        view.nuis_reduce is not EECDFReduce.KEEP
        or view.trans_reduce is not EECDFReduce.KEEP
    )

    if single:
        _plot_quantile_curve(
            ax,
            quantiles[0],
            curves.probs,
            color=color,
            label=label,
            linestyle=linestyle,
            linewidth=options.lw * options.ecdf_lw_scale,
        )
    else:
        if vis_opts.show_bounds:
            lower = np.min(quantiles, axis=0)
            upper = np.max(quantiles, axis=0)

            _plot_quantile_curve(
                ax,
                lower,
                curves.probs,
                color=color,
                label=label,
                linestyle=bounds_linestyle,
                linewidth=options.lw * options.ecdf_lw_scale,
            )
            _plot_quantile_curve(
                ax,
                upper,
                curves.probs,
                color=color,
                linestyle=bounds_linestyle,
                linewidth=options.lw * options.ecdf_lw_scale,
            )

            if options.shade_bounds and not np.array_equal(lower, upper):
                ax.fill_betweenx(
                    np.r_[0.0, curves.probs],
                    np.r_[lower[0], lower],
                    np.r_[upper[0], upper],
                    step="pre",
                    color=color,
                    alpha=options.bounds_alpha,
                    zorder=0.4,
                )

        if vis_opts.centre is not EECDFCentre.NONE:
            centre = (
                np.mean(quantiles, axis=0)
                if vis_opts.centre is EECDFCentre.MEAN
                else np.median(quantiles, axis=0)
            )

            _plot_quantile_curve(
                ax,
                centre,
                curves.probs,
                color=color,
                label=None if vis_opts.show_bounds else label,
                linestyle=linestyle,
                linewidth=options.lw * options.ecdf_lw_scale,
            )

    if profile and show_all and not single:
        for profile_quantiles in quantiles:
            _plot_quantile_curve(
                ax,
                profile_quantiles,
                curves.probs,
                color=color,
                linestyle=bounds_linestyle,
                alpha=options.ecdf_overlay_alpha,
                linewidth=options.lw * options.ecdf_lw_scale,
            )

    # A sole original ECDF is already drawn above. Reduced single curves can
    # still be shown alongside their original selected epistemic case overlays.
    original_single = single and not reduced and len(overlay_indices) == 1

    if not original_single:
        original = curves.orig_quants.reshape(-1, curves.probs.size)

        for overlay_idx, row in enumerate(overlay_indices):
            quantiles = original[row]
            trans_idx, nuis_idx, offset = curves.orig_epis_offsets[row]
            overlay_style = linestyle

            if offset is not EEpisOffset.NOMINAL:
                overlay_style = (
                    options.sim_offset_linestyle
                    if is_sim
                    else options.exp_offset_linestyle
                )

            overlay_label = f"{label} Trans {trans_idx}, Nuis {nuis_idx}"

            if offset is not EEpisOffset.NOMINAL:
                overlay_label += f", {offset.name.lower()} offset"

            if (
                overlay_idx == 0
                and not single
                and not vis_opts.show_bounds
                and vis_opts.centre is EECDFCentre.NONE
            ):
                overlay_label = label
            elif vis_opts.epis_case_idxs is None:
                overlay_label = None

            _plot_quantile_curve(
                ax,
                quantiles,
                curves.probs,
                color=color,
                label=overlay_label,
                linestyle=overlay_style,
                alpha=options.ecdf_overlay_alpha,
                linewidth=options.lw * options.ecdf_lw_scale,
            )


def _plot_nominal_offset_view(
    ax: Axes,
    source: EValidSource,
    curves: ECDFView,
    profile: ECDFView | None,
    options: ValidPlotOpts,
) -> None:
    # Expanded offset order follows EEpisOffset, starting with NOMINAL.
    """Overlay the nominal bounds and centre on a view that spans coherent offsets.

    Parameters
    ----------
    ax : Axes
        Matplotlib axes to draw into. Public plotters create axes when None is
        supplied.
    source : EValidSource
        Simulation or experiment source, used to select labels and styles.
    curves : ECDFView
        Aligned ECDF quantiles and original case metadata for a single physical
        cell.
    profile : ECDFView or None
        Optional nuisance-reduced view used for the nominal centre.
    options : ValidPlotOpts
        Matplotlib styling options used by the plot.
    """
    nominal = curves.quants[:, :, 0].reshape(-1, curves.probs.size)
    is_sim = source is EValidSource.SIM
    color = options.sim_color if is_sim else options.exp_color
    label = options.sim_label if is_sim else options.exp_label

    if nominal.shape[0] > 1:
        _plot_ecdf_limits(
            ax,
            nominal,
            curves.probs,
            color,
            options.sim_bounds_linestyle if is_sim else options.exp_bounds_linestyle,
            f"{label} Epi Env",
            linewidth=options.lw * options.ecdf_lw_scale,
        )

    centre_curves = (
        nominal
        if profile is None
        else profile.quants[:, :, 0].reshape(-1, curves.probs.size)
    )
    _plot_quantile_curve(
        ax,
        np.mean(centre_curves, axis=0),
        curves.probs,
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
    """Draw the pointwise lower and upper curves in an aligned family.

    Parameters
    ----------
    ax : Axes
        Matplotlib axes to draw into. Public plotters create axes when None is
        supplied.
    quantiles : ndarray
        Aligned quantiles of shape (n_curves, n_probs).
    probabilities : ndarray
        One-dimensional cumulative probabilities, with shape (n_probs,).
    color : str
        Matplotlib colour of the plotted line.
    linestyle : _LineStyle
        Matplotlib line style for the displayed curve.
    label : str
        Legend text for the drawn curve or overlay.
    linewidth : float
        Line width in points.
    """
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
    curves: ECDFView,
    alpha: float,
    show_bounds: EShowECDFBounds,
    options: ValidPlotOpts,
) -> None:
    """Draw experimental support limits after per-case confidence shifts.

    Parameters
    ----------
    ax : Axes
        Matplotlib axes to draw into. Public plotters create axes when None is
        supplied.
    curves : ECDFView
        Aligned ECDF quantiles and original case metadata for a single physical
        cell.
    alpha : float
        Experimental confidence parameter used in the Student's t quantile.
    show_bounds : EShowECDFBounds
        Display mode for outer limits or individual offset and case curves.
    options : ValidPlotOpts
        Matplotlib styling options used by the plot.
    """
    if alpha == 0.5:
        return

    shifts = _mavm_confidence_shifts(curves.samples, alpha)[:, None]

    if not np.any(shifts > 0):
        return

    original = curves.orig_quants.reshape(-1, curves.probs.size)
    lower = original - shifts
    upper = original + shifts

    _plot_ecdf_limits(
        ax,
        np.concatenate((lower, upper)),
        curves.probs,
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
                curves.probs,
                color=options.exp_color,
                linestyle=options.exp_conf_linestyle,
                alpha=options.ecdf_overlay_alpha,
                linewidth=options.lw,
            )


def add_support_overlay(
    plot: ECDFPlot,
    supports: ECDFMetricBounds,
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

    Parameters
    ----------
    plot : ECDFPlot
        Existing ECDF plot to modify in place.
    supports : ECDFMetricBounds
        Simulation support bounds with shape ``(*retained_shape, n_probs)``.
    output : EMetricOut or None, optional
        Metric output to select. Use None to infer a single available output.
    selectors : dict[EValidResAxis, int] or None, optional
        Zero-based indices keyed by result axis; omitted axes follow the operation's
        defaults.
    label : str or None, optional
        Legend text for the drawn curve or overlay.
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
        label = (
            "Support"
            if supports.metric is None
            else _create_metric_label(
                supports.metric, metric_labels=options.metric_labels
            )
        )

        if supports.metric is EMetricType.MAVM and output is None:
            label = f"{label} supp"

    for ax, (lower, upper) in zip(plot.axes.flat, prepared, strict=False):
        if np.array_equal(lower, upper):
            continue

        ax.fill_betweenx(
            np.r_[0.0, supports.probs],
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
            supports.probs,
            color=options.metric_line_color,
            linestyle=options.metric_linestyle,
            linewidth=options.lw,
        )
        _plot_quantile_curve(
            ax,
            upper,
            supports.probs,
            color=options.metric_line_color,
            linestyle=options.metric_linestyle,
            linewidth=options.lw,
        )

        ax.legend()
        _apply_plot_options(plot.figure, ax, options, ecdf=True)


def add_normalised_ecdf_axis(
    plot: ECDFPlot,
    reference: float | ECDFReference,
) -> None:
    """Add a secondary x/c axis using one constant per physical panel.

    A reference quantile is evaluated on the displayed family's mean quantile
    curve, after its selected epistemic reductions. This is a constant scaling,
    not a probability-dependent displacement. Metric values retain physical units.

    Parameters
    ----------
    plot : ECDFPlot
        Existing ECDF plot to modify in place.
    reference : float or ECDFReference
        Nonzero scale or reference quantile of a selected source's displayed mean
        curve.
    """

    if plot.normalisation is not None:
        raise ValueError("This plot already has a normalised horizontal axis.")

    constants = []

    if isinstance(reference, ECDFReference):
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
            mean_quantiles = np.mean(curves.quants, axis=(0, 1, 2))
            index = np.searchsorted(curves.probs, probability, side="left")
            constants.append(float(mean_quantiles[index]))
    else:
        if isinstance(reference, (bool, np.bool_)) or not isinstance(
            reference, (int, float, np.integer, np.floating)
        ):
            raise TypeError("Reference must be a constant or ECDFReference.")

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
    comparison: ECDFComparison | None = None,
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

    Parameters
    ----------
    plot : ECDFPlot
        Existing ECDF plot to modify in place.
    metric : IValidMetric
        Metric identifier or strategy used by the plot or overlay.
    output : EMetricOut or None, optional
        Metric output to select. Use None to infer a single available output.
    display : EMetricDisp or None, optional
        Overlay geometry to draw. None chooses a default for the metric.
    comparison : ECDFComparison or None, optional
        Optional explicit original simulation and experiment case pair.
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
        if not isinstance(comparison, ECDFComparison):
            raise TypeError("Comparison must be ECDFComparison.")

        for index in (
            comparison.sim_trans_idx,
            comparison.sim_nuis_idx,
            comparison.exp_trans_idx,
            comparison.exp_nuis_idx,
        ):
            if isinstance(index, (bool, np.bool_)) or not isinstance(
                index, (int, np.integer)
            ):
                raise TypeError("Comparison case indices must be integers.")

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
                sim_key not in sim.orig_epis_offsets
                or exp_key not in exp.orig_epis_offsets
            ):
                raise ValueError(
                    "Comparison is outside the plotted epistemic case selections."
                )

            sim_indices = (sim.orig_epis_offsets.index(sim_key),)
            exp_indices = (exp.orig_epis_offsets.index(exp_key),)

        if display is EMetricDisp.INTEGRATION_AREA and (
            len(sim_indices) != 1 or len(exp_indices) != 1
        ):
            raise ValueError(
                "Integration shading requires an explicit original sim–exp case pair."
            )

        value_columns = (
            len(output_indices)
            if display is EMetricDisp.SUPPORT_SHIFT
            else len(exp_indices)
        )
        values = np.full((len(sim_indices), value_columns), np.nan)

        # KS envelope geometry is calculated directly from displayed bounds;
        # it is not claimed to be the maximum KS over original sim–exp case pairs.
        if display is not EMetricDisp.CDF_DIFFERENCE:
            for row, sim_index in enumerate(sim_indices):
                for column, exp_index in enumerate(exp_indices):
                    result = metric.calc(sim.samples[sim_index], exp.samples[exp_index])
                    metric_values = extract_metric_outputs(result)

                    if display is EMetricDisp.SUPPORT_SHIFT:
                        for direction_idx, output_idx in enumerate(output_indices):
                            values[row, direction_idx] = np.fmax(
                                values[row, direction_idx], metric_values[output_idx]
                            )
                    else:
                        values[row, column] = metric_values[output_indices[0]]

        prepared.append((sim, exp, sim_indices, exp_indices, values))

    options = plot.options
    metric_name = _create_metric_label(kind, metric_labels=options.metric_labels)
    metric_label = _create_metric_label(
        kind, output, metric_labels=options.metric_labels
    )

    for ax, (sim, exp, sim_indices, exp_indices, values) in zip(
        plot.axes.flat, prepared, strict=False
    ):
        annotation = None

        if display is EMetricDisp.SUPPORT_SHIFT:
            quantiles = sim.orig_quants.reshape(-1, sim.probs.size)[list(sim_indices)]
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
                np.r_[0.0, sim.probs],
                np.r_[lower[0], lower],
                np.r_[upper[0], upper],
                step="pre",
                color=options.metric_color,
                alpha=options.metric_alpha,
                zorder=0.5,
                label=f"{metric_name} supp" if full_support else metric_name,
            )
            _plot_quantile_curve(
                ax,
                lower,
                sim.probs,
                color=options.metric_line_color,
                linestyle=options.metric_linestyle,
                linewidth=options.lw,
            )
            _plot_quantile_curve(
                ax,
                upper,
                sim.probs,
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
                exp_cdf = ECDF(exp_bound, mavm.exp_probs)

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
                np.r_[sim_cdf.probs, exp_cdf.probs]
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
                label=metric_label if kind is EMetricType.MAVM else metric_name,
            )
        elif display is EMetricDisp.CDF_DIFFERENCE:
            if comparison is not None or len(sim_indices) == len(exp_indices) == 1:
                curve_pairs = (
                    (
                        sim.orig_quants.reshape(-1, sim.probs.size)[sim_indices[0]],
                        exp.orig_quants.reshape(-1, exp.probs.size)[exp_indices[0]],
                    ),
                )
            else:
                sim_quantiles = sim.quants.reshape(-1, sim.probs.size)
                exp_quantiles = exp.quants.reshape(-1, exp.probs.size)
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

            gaps = []

            for pair_index, (sim_quantiles, exp_quantiles) in enumerate(curve_pairs):
                support = np.unique(np.r_[sim_quantiles, exp_quantiles])
                sim_probabilities = np.r_[0.0, sim.probs][
                    np.searchsorted(sim_quantiles, support, side="right")
                ]
                exp_probabilities = np.r_[0.0, exp.probs][
                    np.searchsorted(exp_quantiles, support, side="right")
                ]

                differences = np.abs(sim_probabilities - exp_probabilities)
                index = int(np.argmax(differences))
                gaps.append(differences[index])

                ax.plot(
                    [support[index], support[index]],
                    [sim_probabilities[index], exp_probabilities[index]],
                    color=options.metric_line_color,
                    linestyle=options.metric_linestyle,
                    marker="_",
                    label=metric_name if pair_index == 0 else None,
                    linewidth=options.lw,
                )

            annotation = f"{metric_name} = {gaps[0]:.2f}"

            if len(gaps) == 2:
                annotation += f" / {gaps[1]:.2f} (lower / upper)"
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
            annotation = f"{label} [{units}]"

        if annotation is not None:
            # Place later annotations below earlier ones, preserving composability.
            ax.text(
                0.98,
                0.98 - 0.07 * len(ax.texts),
                annotation,
                transform=ax.transAxes,
                ha="right",
                va="top",
                fontsize=options.font_leg_size,
            )

        if ax.get_legend_handles_labels()[1]:
            ax.legend()

        _apply_plot_options(plot.figure, ax, options, ecdf=True)


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
    """Plot a selected epistemic case over time. MAVM defaults to both directions.

    Parameters
    ----------
    result : ValidResult
        Validation results containing the requested observable and metric.
    observable : str
        Observable tag identifying matching metadata and observations.
    metric : EMetricType
        Metric identifier or strategy used by the plot or overlay.
    selectors : dict[EValidResAxis, int] or None, optional
        Zero-based indices keyed by result axis; omitted axes follow the operation's
        defaults.
    output : EMetricOut or None, optional
        Metric output to select. Use None to infer a single available output.
    valid_plot_opts : ValidPlotOpts or None, optional
        Matplotlib layout and appearance settings. Use None for defaults.
    ax : Axes or None, optional
        Matplotlib axes to draw into. Public plotters create axes when None is
        supplied.

    Returns
    -------
    tuple[Figure, Axes]
        Figure and axes with metric values against the one-dimensional physical time
        coordinate.
    """
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
        trace_label = _create_metric_label(
            metric,
            direction if metric is EMetricType.MAVM else None,
            metric_labels=options.metric_labels,
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

    label = _create_metric_label(
        metric,
        outputs[0] if len(outputs) == 1 else None,
        metric_labels=options.metric_labels,
    )
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
    """Histogram finite metric values over selected comparison cells.

    Parameters
    ----------
    result : ValidResult
        Validation results containing the requested observable and metric.
    observable : str
        Observable tag identifying matching metadata and observations.
    metric : EMetricType
        Metric identifier or strategy used by the plot or overlay.
    output : EMetricOut or None, optional
        Metric output to select. Use None to infer a single available output.
    selectors : dict[EValidResAxis, int] or None, optional
        Zero-based indices keyed by result axis; omitted axes follow the operation's
        defaults.
    valid_plot_opts : ValidPlotOpts or None, optional
        Matplotlib layout and appearance settings. Use None for defaults.
    ax : Axes or None, optional
        Matplotlib axes to draw into. Public plotters create axes when None is
        supplied.

    Returns
    -------
    tuple[Figure, Axes]
        Figure and axes with a histogram of the selected finite metric cells.
    """
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

    metric_label = _create_metric_label(
        metric, output, metric_labels=options.metric_labels
    )
    ax.set_xlabel(f"{metric_label} [{_metric_units(result, observable, metric)}]")
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

    Parameters
    ----------
    point_count : int
        Number of visible panels.
    rows : int or None, optional
        Requested grid rows. None chooses the count from the panel count and
        columns.
    columns : int or None, optional
        Requested grid columns. None chooses the count automatically.
    valid_plot_opts : ValidPlotOpts or None, optional
        Matplotlib layout and appearance settings. Use None for defaults.

    Returns
    -------
    tuple[Figure, ndarray]
        Figure and object array of Matplotlib axes with shape (n_rows, n_columns).
        Spare axes are hidden.
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
    columns: int | None = None,
    valid_plot_opts: ValidPlotOpts | None = None,
) -> tuple[Figure, np.ndarray]:
    """Plot a selected epistemic case per point. MAVM defaults to both directions.

    The panel grid is sized from the point count unless columns is supplied.

    Parameters
    ----------
    result : ValidResult
        Validation results containing the requested observable and metric.
    observable : str
        Observable tag identifying matching metadata and observations.
    metric : EMetricType
        Metric identifier or strategy used by the plot or overlay.
    point_idxs : tuple[int, ...]
        Original point indices in panel order. None selects all available points.
    selectors : dict[EValidResAxis, int] or None, optional
        Zero-based indices keyed by result axis; omitted axes follow the operation's
        defaults.
    output : EMetricOut or None, optional
        Metric output to select. Use None to infer a single available output.
    columns : int or None, optional
        Requested grid columns. None chooses the count automatically.
    valid_plot_opts : ValidPlotOpts or None, optional
        Matplotlib layout and appearance settings. Use None for defaults.

    Returns
    -------
    tuple[Figure, ndarray]
        Figure and two-dimensional axes grid with one time-series panel per selected
        point.
    """
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

    Parameters
    ----------
    result : ValidResult
        Validation results containing the requested observable and metric.
    observable : str
        Observable tag identifying matching metadata and observations.
    metric : EMetricType
        Metric identifier or strategy used by the plot or overlay.
    profile : ValidMetricExtrema
        Previously computed metric extrema and their controlling case indices.
    point_idxs : tuple[int, ...] or None, optional
        Original point indices in panel order. None selects all available points.
    selectors : dict[EValidResAxis, int] or None, optional
        Zero-based indices keyed by result axis; omitted axes follow the operation's
        defaults.
    output : EMetricOut or None, optional
        Metric output to select. Use None to infer a single available output.
    columns : int, optional
        Requested grid columns. None chooses the count automatically.
    valid_plot_opts : ValidPlotOpts or None, optional
        Matplotlib layout and appearance settings. Use None for defaults.

    Returns
    -------
    tuple[Figure, ndarray]
        Figure and two-dimensional axes grid with lower and upper metric traces per
        selected point.
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
    metric_label = _create_metric_label(
        metric,
        outputs[0] if len(outputs) == 1 else None,
        metric_labels=options.metric_labels,
    )

    for point_idx, point_traces, ax in zip(points, traces, axes.flat, strict=False):
        for direction, (lower, upper) in zip(outputs, point_traces, strict=True):
            label = _create_metric_label(
                metric,
                direction if metric is EMetricType.MAVM else None,
                metric_labels=options.metric_labels,
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

    Parameters
    ----------
    valid_case : ValidCase
        Aligned simulation and experiment observations with their named axes.
    observable : str
        Observable tag identifying matching metadata and observations.
    metric : EMetricType
        Metric identifier or strategy used by the plot or overlay.
    profile : ValidMetricExtrema
        Previously computed metric extrema and their controlling case indices.
    selectors : dict[EValidResAxis, int] or None, optional
        Zero-based indices keyed by result axis; omitted axes follow the operation's
        defaults.
    output : EMetricOut or None, optional
        Metric output to select. Use None to infer a single available output.
    valid_plot_opts : ValidPlotOpts or None, optional
        Matplotlib layout and appearance settings. Use None for defaults.

    Returns
    -------
    tuple[Figure, ndarray]
        Figure and axes array of shape (1, 2), showing the original minimum and
        maximum comparisons.
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

    options = valid_plot_opts or ValidPlotOpts()
    fig, axes = create_valid_plot_grid(2, rows=1, valid_plot_opts=options)
    label = _create_metric_label(
        metric,
        output if metric is EMetricType.MAVM else None,
        metric_labels=options.metric_labels,
    )

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
            sim_view=ECDFViewSel(trans_idxs=(sim_trans,), nuis_idxs=(sim_nuis,)),
            exp_view=ECDFViewSel(trans_idxs=(exp_trans,), nuis_idxs=(exp_nuis,)),
            sim_offset=sim_offset,
            exp_offset=exp_offset,
            point_idxs=(comparison_indices.get(EValidResAxis.POINT, 0),),
            component_idx=comparison_indices.get(EValidResAxis.COMPONENT, 0),
            time_idx=comparison_indices.get(EValidResAxis.TIME, 0),
            valid_plot_opts=options,
            ax=ax,
        )

        sim_case = f"Sim. trans. {sim_trans}, nuis. {sim_nuis}"
        exp_case = f"Exp. trans. {exp_trans}, nuis. {exp_nuis}"

        if EValidResAxis.SIM_OFFSET in comparison_indices:
            sim_case += f", {sim_offset.name.lower()} offset"

        if EValidResAxis.EXP_OFFSET in comparison_indices:
            exp_case += f", {exp_offset.name.lower()} offset"

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
    """Draw a single empirical quantile curve as a probability step plot.

    Parameters
    ----------
    ax : Axes
        Matplotlib axes to draw into. Public plotters create axes when None is
        supplied.
    quantiles : ndarray
        Curve coordinates of shape (n_probs,), in observable units.
    probabilities : ndarray
        One-dimensional cumulative probabilities, with shape (n_probs,).
    color : str
        Matplotlib colour of the plotted line.
    label : str or None, optional
        Legend text for the drawn curve or overlay.
    linestyle : _LineStyle
        Matplotlib line style for the displayed curve.
    linewidth : float
        Line width in points.
    alpha : float, optional
        Opacity of the plotted curve.
    """
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


def _plot_distribution_bounds(
    ax: Axes,
    curves: ECDFView,
    support_cases: tuple[int, int] | None,
    *,
    color: str,
    label: str,
    options: ValidPlotOpts,
    bounds_linestyle: _LineStyle,
    linestyle: _LineStyle,
) -> None:
    """Draw a mean quantile curve and selected distribution endpoints.

    Parameters
    ----------
    ax : Axes
        Matplotlib axes to draw into. Public plotters create axes when None is
        supplied.
    curves : ECDFView
        Aligned ECDF quantiles and original case metadata for a single physical
        cell.
    support_cases : tuple[int, int] or None
        Original flat indices for the lower and upper whole-case curves. None uses
        pointwise limits.
    color : str
        Matplotlib colour of the plotted line.
    label : str
        Legend text for the drawn curve or overlay.
    options : ValidPlotOpts
        Matplotlib styling options used by the plot.
    bounds_linestyle : _LineStyle
        Line style used for lower and upper bound curves.
    linestyle : _LineStyle
        Matplotlib line style for the displayed curve.
    """
    if support_cases is None:
        bounds = calc_ecdf_epis_bounds(curves)
        lower, upper = bounds.lower, bounds.upper
    else:
        original = curves.orig_quants.reshape(-1, curves.probs.size)
        lower, upper = original[list(support_cases)]

    _plot_quantile_curve(
        ax,
        np.mean(curves.quants, axis=(0, 1, 2)),
        curves.probs,
        color=color,
        label=label,
        linestyle=linestyle,
        linewidth=options.lw * options.ecdf_lw_scale,
    )
    _plot_quantile_curve(
        ax,
        lower,
        curves.probs,
        color=color,
        linestyle=bounds_linestyle,
        linewidth=options.lw * options.ecdf_lw_scale,
    )
    _plot_quantile_curve(
        ax,
        upper,
        curves.probs,
        color=color,
        linestyle=bounds_linestyle,
        linewidth=options.lw * options.ecdf_lw_scale,
    )

    if options.shade_bounds and not np.array_equal(lower, upper):
        ax.fill_betweenx(
            np.r_[0.0, curves.probs],
            np.r_[lower[0], lower],
            np.r_[upper[0], upper],
            step="pre",
            color=color,
            alpha=options.bounds_alpha,
            zorder=0.4,
        )


def plot_valid_mavm_diagnostic(
    sim: ECDFView,
    exp: ECDFView | None = None,
    *,
    sim_support_cases: tuple[int, int] | None = None,
    exp_support_cases: tuple[int, int] | None = None,
    bound_pairs: MAVMBoundPairs | None = None,
    view: EValidMAVMDiagnostic = EValidMAVMDiagnostic.DISTRIBUTIONS,
    valid_plot_opts: ValidPlotOpts | None = None,
    ax: Axes | None = None,
) -> tuple[Figure, Axes]:
    """Compare uncertainty bands, shifted endpoint curves or MAVM support.

    Support case indices select whole original curves. With no indices,
    the diagnostic draws pointwise bounds.

    The bound pairs provide the scalar MAVM distances used to shift sim
    support and draw the outer support band.

    Parameters
    ----------
    sim : ECDFView
        Computed simulation ECDF view.
    exp : ECDFView or None, optional
        Optional experimental ECDF view.
    sim_support_cases : tuple[int, int] or None, optional
        Optional pair of original simulation case indices for the lower and upper
        curves.
    exp_support_cases : tuple[int, int] or None, optional
        Optional pair of original experimental case indices for the lower and upper
        curves.
    bound_pairs : MAVMBoundPairs or None, optional
        MAVM endpoint comparisons required for shifted support and comparison views.
    view : EValidMAVMDiagnostic, optional
        Diagnostic mode for distributions, comparisons or corrected support.
    valid_plot_opts : ValidPlotOpts or None, optional
        Matplotlib layout and appearance settings. Use None for defaults.
    ax : Axes or None, optional
        Matplotlib axes to draw into. Public plotters create axes when None is
        supplied.

    Returns
    -------
    tuple[Figure, Axes]
        Figure and axes with the selected MAVM diagnostic.
    """

    if not isinstance(view, EValidMAVMDiagnostic):
        raise TypeError("Diagnostic view must be EValidMAVMDiagnostic.")

    if view is not EValidMAVMDiagnostic.DISTRIBUTIONS and bound_pairs is None:
        raise ValueError("Selected diagnostic requires MAVM bound pairs.")

    fig, ax = _create_axes(ax, valid_plot_opts)
    options = ValidPlotOpts() if valid_plot_opts is None else valid_plot_opts

    if exp is not None and view is not EValidMAVMDiagnostic.MODEL_CORRECTION:
        _plot_distribution_bounds(
            ax,
            exp,
            exp_support_cases,
            color=options.exp_color,
            label=options.exp_label,
            options=options,
            linestyle=options.exp_linestyle,
            bounds_linestyle=options.exp_bounds_linestyle,
        )

    _plot_distribution_bounds(
        ax,
        sim,
        sim_support_cases,
        color=options.sim_color,
        label=options.sim_label,
        options=options,
        linestyle=options.sim_linestyle,
        bounds_linestyle=options.sim_bounds_linestyle,
    )

    if bound_pairs is not None and view is EValidMAVMDiagnostic.COMPARISONS:
        for comparison, linestyle, label in (
            (bound_pairs.upper, options.metric_linestyle, "Upper case"),
            (bound_pairs.lower, options.metric_alt_linestyle, "Lower case"),
        ):
            _plot_quantile_curve(
                ax,
                comparison.sim_quantiles + comparison.d_plus,
                comparison.sim_probs,
                color=options.metric_line_color,
                linestyle=linestyle,
                label=label,
                linewidth=options.lw,
            )
            _plot_quantile_curve(
                ax,
                comparison.sim_quantiles - comparison.d_minus,
                comparison.sim_probs,
                color=options.metric_line_color,
                linestyle=linestyle,
                linewidth=options.lw,
            )

    if bound_pairs is not None and view in (
        EValidMAVMDiagnostic.CORRECTION,
        EValidMAVMDiagnostic.MODEL_CORRECTION,
    ):
        lower_cdf = ECDF(
            bound_pairs.lower.sim_quantiles - bound_pairs.lower.d_minus,
            bound_pairs.lower.sim_probs,
        )
        upper_cdf = ECDF(
            bound_pairs.upper.sim_quantiles + bound_pairs.upper.d_plus,
            bound_pairs.upper.sim_probs,
        )

        probabilities = np.unique(
            np.r_[lower_cdf.probs, upper_cdf.probs]
        )
        lower = _evaluate_ecdf_quantiles(lower_cdf, probabilities)
        upper = _evaluate_ecdf_quantiles(upper_cdf, probabilities)

        metric_name = _create_metric_label(
            EMetricType.MAVM, metric_labels=options.metric_labels
        )
        support_label = f"{metric_name} supp"

        if options.shade_bounds:
            ax.fill_betweenx(
                np.r_[0.0, probabilities],
                np.r_[lower[0], lower],
                np.r_[upper[0], upper],
                step="pre",
                color=options.metric_color,
                alpha=options.metric_alpha,
                zorder=0.5,
                label=support_label,
            )

        _plot_quantile_curve(
            ax,
            lower,
            probabilities,
            color=options.metric_line_color,
            linestyle=options.metric_linestyle,
            label=None if options.shade_bounds else support_label,
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
