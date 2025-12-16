# ==============================================================================
# pyvale: the python validation engine
# License: MIT
# Copyright (C) 2025 The Computer Aided Validation Team
# ==============================================================================

"""
This module contains functions for plotting virtuals sensor trace summary
statistics and uncertainty bounds over simulated experiments.
"""

from typing import Any
import numpy as np
import matplotlib.figure as mpf
import matplotlib.axes._axes as mpa
import matplotlib.pyplot as plt
from pyvale.sensorsim.exceptions import VisError
from pyvale.sensorsim.visualopts import (PlotOptsGeneral,
                                         TraceOptsExperiment,
                                         EExpVisBounds,
                                         EExpVisCentre)
from pyvale.sensorsim.sensordescriptor import SensorDescriptor
from pyvale.sensorsim.experimentsimulator import (ExperimentSimulator,
                                                  ExpSimSaveKeys)
from pyvale.sensorsim.experimentstats import calc_sensor_array_stats


def plot_exp_traces(exp_data: dict[tuple[str,...],np.ndarray],
                    comp_ind: int,
                    sens_key: str,
                    sim_key: str,
                    descriptor: SensorDescriptor,
                    trace_opts: TraceOptsExperiment | None = None,
                    plot_opts: PlotOptsGeneral | None = None,
                    exp_save_keys: ExpSimSaveKeys | None = None,
                    ) -> tuple[mpf.Figure,mpa.Axes]:
    """Plots the traces from a set of simulated experiments for a given input
    physics simulation and sensor array.

    Parameters
    ----------
    exp_data : dict[tuple[str,...],np.ndarray]
        Simulated experiment data dictionary produced by the experiment
        simulator.
    comp_ind : int
        Index for the component of the measurement array to plot. For scalar
        field sensors there is only one component so this should be 0. For
        vector field sensors the components are in the order the keys where
        specified which is normally x,y,z. For tensor field sensors the keys are
        in the order they are specified with normal followed by deviatoric
        components.
    sens_key : str
        String key identifying the sensor array in the simulated experiment data
        dictionary.
    sim_key : str
        String key identifying the input physics simulation in the simulation
        data dictionary.
    descriptor : SensorDescriptor
        Descriptor containing strings for labelling the plot axes with sensor
        units and names.
    trace_opts : TraceOptsExperiment | None, optional
        Options for plotting the experiment traces including how to display the
        scatter in the traces, by default None. If None a default options
        dataclass is created.
    plot_opts : PlotOptsGeneral | None, optional
        Options for controlling characteristics of the plot including the size
        of the figure, line widths etc., by default None. If None a default
        plot options dataclass is created.
    exp_save_keys : ExpSimSaveKeys | None, optional
        Keys for extracting the simulation data from the simulated experiment
        data dictionary, by default None. If None the default keys are used.

    Returns
    -------
    tuple[mpf.Figure,mpa.Axes]
        Figure and axes object for the matplotlib plot that is created.
    """
    if trace_opts is None:
        trace_opts = TraceOptsExperiment()

    if plot_opts is None:
        plot_opts = PlotOptsGeneral()

    if exp_save_keys is None:
        exp_save_keys = ExpSimSaveKeys()

    meas_key = (sim_key,sens_key,exp_save_keys.meas)
    sys_key = (sim_key,sens_key,exp_save_keys.sys)
    rand_key = (sim_key,sens_key,exp_save_keys.rand)
    time_key = (sim_key,sens_key,exp_save_keys.sens_times)

    exp_arr = exp_data[meas_key]
    samp_time = exp_data[time_key]

    num_exp_per_sim = exp_arr.shape[0]
    num_sens = exp_arr.shape[1]

    if trace_opts.sensors_to_plot is None:
        sensors_to_plot = range(num_sens)
    else:
        sensors_to_plot = trace_opts.sensors_to_plot

    #---------------------------------------------------------------------------
    # Figure canvas setup
    (fig, ax) = plt.subplots(figsize=plot_opts.single_fig_size_landscape,
                           layout='constrained')
    fig.set_dpi(plot_opts.resolution)

    #---------------------------------------------------------------------------
    # Plot all simulated experimental points
    if trace_opts.plot_all_exp_points:
        for ss in sensors_to_plot:
            for ee in range(num_exp_per_sim):
                ax.plot(samp_time,
                        exp_arr[ee,ss,comp_ind,:],
                        "+",
                        lw=plot_opts.lw,
                        ms=plot_opts.ms,
                        color=plot_opts.colors[ss % plot_opts.colors_num])

    sensor_tags = descriptor.create_sensor_tags(num_sens)
    lines = []

    # TODO: limit this to only calculate what we need for the fill and centre
    exp_stats = calc_sensor_array_stats(exp_arr)

    for ss in sensors_to_plot:
        if trace_opts.centre == EExpVisCentre.MEDIAN:
            trace_centre = exp_stats.med[ss,comp_ind,:]
        else:
            trace_centre = exp_stats.mean[ss,comp_ind,:]

        line, = ax.plot(samp_time,
                trace_centre,
                trace_opts.exp_centre_line,
                label=sensor_tags[ss],
                lw=plot_opts.lw,
                ms=plot_opts.ms,
                color=plot_opts.colors[ss % plot_opts.colors_num])
        lines.append(line)

        if trace_opts.fill_between is not None:
            if trace_opts.fill_between == EExpVisBounds.MINMAX:
                upper = trace_opts.fill_scale*exp_stats.min
                lower = trace_opts.fill_scale*exp_stats.max
            elif trace_opts.fill_between == EExpVisBounds.QUARTILE:
                upper = trace_opts.fill_scale*exp_stats.q25
                lower = trace_opts.fill_scale*exp_stats.q75
            elif trace_opts.fill_between == EExpVisBounds.STD:
                upper = trace_centre + \
                        trace_opts.fill_scale*exp_stats.std
                lower = trace_centre - \
                        trace_opts.fill_scale*exp_stats.std
            elif trace_opts.fill_between == EExpVisBounds.MAD:
                upper = trace_centre + \
                        trace_opts.fill_scale*exp_stats.mad
                lower = trace_centre - \
                        trace_opts.fill_scale*exp_stats.mad

            ax.fill_between(samp_time,
                upper[ss,comp_ind,:],
                lower[ss,comp_ind,:],
                color=plot_opts.colors[ss % plot_opts.colors_num],
                alpha=0.2)

    #---------------------------------------------------------------------------
    # Plot simulation and truth line
    if (trace_opts.truth_line is not None 
        and sys_key in exp_data 
        and rand_key in exp_data):
        truth = exp_data[meas_key] - exp_data[sys_key] - exp_data[rand_key]
        truth = truth[0,:,:,:]
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
