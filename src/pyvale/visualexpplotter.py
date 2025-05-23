# ==============================================================================
# pyvale: the python validation engine
# License: MIT
# Copyright (C) 2025 The Computer Aided Validation Team
# ==============================================================================

"""
This module contains function for plotting virtuals sensor trace summary
statistics and uncertainty bounds over simulated experiments.
"""

from typing import Any
import numpy as np
import matplotlib.pyplot as plt
from pyvale.pyvaleexceptions import VisError
from pyvale.visualopts import (PlotOptsGeneral,
                               TraceOptsExperiment,
                               EExpVisBounds,
                               EExpVisCentre)
from pyvale.experimentsimulator import ExperimentSimulator


def plot_exp_traces(exp_sim: ExperimentSimulator,
                    component: str,
                    sens_array_num: int,
                    sim_num: int,
                    trace_opts: TraceOptsExperiment | None = None,
                    plot_opts: PlotOptsGeneral | None = None) -> tuple[Any,Any]:
    """Plots time traces for summary statistics of virtual sensor traces over
    a series of virtual experiments.

    Parameters
    ----------
    exp_sim : ExperimentSimulator
        Experiment simulation object containing the set of virtual experiment to
        be plotted.
    component : str
        String key for the field component to plot.
    sens_array_num : int
        List index for the sensor array to plot.
    sim_num : int
        Index for the simulation to plot.
    trace_opts : TraceOptsExperiment | None, optional
        Dataclass containing specific options for controlling the plot
        appearance, by default None. If None the default options are used.
    plot_opts : PlotOptsGeneral | None, optional
        Dataclass containing general options for formatting plots and
        visualisations, by default None. If None the default options are used.

    Returns
    -------
    tuple[Any,Any]
        A tuple containing a handle to the matplotlib figure and axis objects:
        (fig,ax).

    Raises
    ------
    VisError
        There are no virtual experiments or virtuale experiment stats to plot in
        the ExperimentSimulator object. Call 'run_experiments' and 'calc_stats'.
    """
    if trace_opts is None:
        trace_opts = TraceOptsExperiment()

    if plot_opts is None:
        plot_opts = PlotOptsGeneral()

    descriptor = exp_sim._sensor_arrays[sens_array_num]._descriptor
    comp_ind = exp_sim._sensor_arrays[sens_array_num].get_field().get_component_index(component)
    samp_time = exp_sim._sensor_arrays[sens_array_num].get_sample_times()
    num_sens = exp_sim._sensor_arrays[sens_array_num].get_measurement_shape()[0]

    exp_data = exp_sim._exp_data
    exp_stats = exp_sim._exp_stats

    if exp_data is None or exp_stats is None:
        raise VisError("Before visualising virtual experiment traces the " \
        "virtual experiments must be run. exp_data or exp_stats is None.")

    if trace_opts.sensors_to_plot is None:
        sensors_to_plot = range(num_sens)
    else:
        sensors_to_plot = trace_opts.sensors_to_plot

    #---------------------------------------------------------------------------
    # Figure canvas setup
    fig, ax = plt.subplots(figsize=plot_opts.single_fig_size_landscape,
                           layout='constrained')
    fig.set_dpi(plot_opts.resolution)

    #---------------------------------------------------------------------------
    # Plot all simulated experimental points
    if trace_opts.plot_all_exp_points:
        for ss in sensors_to_plot:
            for ee in range(exp_sim._num_exp_per_sim):
                ax.plot(samp_time,
                        exp_data[sens_array_num][sim_num,ee,ss,comp_ind,:],
                        "+",
                        lw=plot_opts.lw,
                        ms=plot_opts.ms,
                        color=plot_opts.colors[ss % plot_opts.colors_num])

    sensor_tags = descriptor.create_sensor_tags(num_sens)
    lines = []
    for ss in sensors_to_plot:
        if trace_opts.centre == EExpVisCentre.MEDIAN:
            trace_centre = exp_stats[sens_array_num].med[sim_num,ss,comp_ind,:]
        else:
            trace_centre = exp_stats[sens_array_num].mean[sim_num,ss,comp_ind,:]

        line, = ax.plot(samp_time,
                trace_centre,
                trace_opts.exp_centre_line,
                label=sensor_tags[ss],
                lw=plot_opts.lw,
                ms=plot_opts.ms,
                color=plot_opts.colors[ss % plot_opts.colors_num])
        lines.append(line)

        if trace_opts.fill_between is not None:
            upper = np.zeros_like(exp_stats[sens_array_num].min)
            lower = np.zeros_like(exp_stats[sens_array_num].min)

            if trace_opts.fill_between == EExpVisBounds.MINMAX:
                upper = trace_opts.fill_scale*exp_stats[sens_array_num].min
                lower = trace_opts.fill_scale*exp_stats[sens_array_num].max
            elif trace_opts.fill_between == EExpVisBounds.QUARTILE:
                upper = trace_opts.fill_scale*exp_stats[sens_array_num].q25
                lower = trace_opts.fill_scale*exp_stats[sens_array_num].q75
            elif trace_opts.fill_between == EExpVisBounds.STD:
                upper = trace_centre + \
                        trace_opts.fill_scale*exp_stats[sens_array_num].std
                lower = trace_centre - \
                        trace_opts.fill_scale*exp_stats[sens_array_num].std
            elif trace_opts.fill_between == EExpVisBounds.MAD:
                upper = trace_centre + \
                        trace_opts.fill_scale*exp_stats[sens_array_num].mad
                lower = trace_centre - \
                        trace_opts.fill_scale*exp_stats[sens_array_num].mad

            ax.fill_between(samp_time,
                upper[sim_num,ss,comp_ind,:],
                lower[sim_num,ss,comp_ind,:],
                color=plot_opts.colors[ss % plot_opts.colors_num],
                alpha=0.2)

    #---------------------------------------------------------------------------
    # Plot simulation and truth line
    if trace_opts.sim_line is not None:
        sim_time = exp_sim._sensor_arrays[sens_array_num].get_field().get_time_steps()
        sim_vals = exp_sim._sensor_arrays[sens_array_num].get_field().sample_field(
                   exp_sim._sensor_arrays[sens_array_num]._positions)

        for ss in sensors_to_plot:
            ax.plot(sim_time,
                    sim_vals[ss,comp_ind,:],
                    trace_opts.sim_line,
                    lw=plot_opts.lw,
                    ms=plot_opts.ms)

    if trace_opts.truth_line is not None:
        truth = exp_sim._sensor_arrays[sens_array_num].get_truth()
        for ss in sensors_to_plot:
            ax.plot(samp_time,
                    truth[ss,comp_ind,:],
                    trace_opts.truth_line,
                    lw=plot_opts.lw,
                    ms=plot_opts.ms,
                    color=plot_opts.colors[ss % plot_opts.colors_num])

    #---------------------------------------------------------------------------
    # Axis / legend labels and options
    ax.set_xlabel(trace_opts.time_label,
                fontsize=plot_opts.font_ax_size, fontname=plot_opts.font_name)
    ax.set_ylabel(descriptor.create_label(comp_ind),
                fontsize=plot_opts.font_ax_size, fontname=plot_opts.font_name)

    if trace_opts.time_min_max is None:
        ax.set_xlim((np.min(samp_time),np.max(samp_time))) # type: ignore
    else:
        ax.set_xlim(trace_opts.time_min_max)

    if trace_opts.legend_loc is not None:
        ax.legend(handles=lines,
                  prop={"size":plot_opts.font_leg_size},
                  loc=trace_opts.legend_loc)

    plt.grid(True)
    plt.draw()

    return (fig,ax)