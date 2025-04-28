# ================================================================================
# pyvale: the python validation engine
# License: MIT
# Copyright (C) 2025 The Computer Aided Validation Team
# ================================================================================

from typing import Any

import numpy as np
import matplotlib.pyplot as plt
from pyvale.sensorarraypoint import SensorArrayPoint
from pyvale.visualopts import (PlotOptsGeneral,
                                    TraceOptsSensor,
                                    TraceOptsExperiment)
from pyvale.experimentsimulator import ExperimentSimulator

#TODO: Docstrings

def plot_time_traces(sensor_array: SensorArrayPoint,
                     component: str,
                     trace_opts: TraceOptsSensor | None = None,
                     plot_opts: PlotOptsGeneral | None = None
                     ) -> tuple[Any,Any]:

    #---------------------------------------------------------------------------
    field = sensor_array.field
    comp_ind = sensor_array.field.get_component_index(component)
    samp_time = sensor_array.get_sample_times()
    measurements = sensor_array.get_measurements()
    n_sensors = sensor_array.sensor_data.positions.shape[0]
    descriptor = sensor_array.descriptor
    sensors_perturbed = sensor_array.get_sensor_data_perturbed()

    #---------------------------------------------------------------------------
    if plot_opts is None:
        plot_opts = PlotOptsGeneral()

    if trace_opts is None:
        trace_opts = TraceOptsSensor()

    if trace_opts.sensors_to_plot is None:
        sensors_to_plot = range(n_sensors)
    else:
        sensors_to_plot = trace_opts.sensors_to_plot

    #---------------------------------------------------------------------------
    # Figure canvas setup
    fig, ax = plt.subplots(figsize=plot_opts.single_fig_size_landscape,
                           layout='constrained')
    fig.set_dpi(plot_opts.resolution)

    #---------------------------------------------------------------------------
    # Plot simulation and truth lines
    if trace_opts.sim_line is not None:
        sim_time = field.get_time_steps()
        sim_vals = field.sample_field(sensor_array.sensor_data.positions,
                                      None,
                                      sensor_array.sensor_data.angles)

        for ii,ss in enumerate(sensors_to_plot):
            ax.plot(sim_time,
                    sim_vals[ss,comp_ind,:],
                    trace_opts.sim_line,
                    lw=plot_opts.lw,
                    ms=plot_opts.ms,
                    color=plot_opts.colors[ii % plot_opts.n_colors])

    if trace_opts.truth_line is not None:
        truth = sensor_array.get_truth()
        for ii,ss in enumerate(sensors_to_plot):
            ax.plot(samp_time,
                    truth[ss,comp_ind,:],
                    trace_opts.truth_line,
                    lw=plot_opts.lw,
                    ms=plot_opts.ms,
                    color=plot_opts.colors[ii % plot_opts.n_colors])

    sensor_tags = descriptor.create_sensor_tags(n_sensors)
    for ii,ss in enumerate(sensors_to_plot):
        sensor_time = samp_time
        if sensors_perturbed is not None:
            if sensors_perturbed.sample_times is not None:
                sensor_time = sensors_perturbed.sample_times

        ax.plot(sensor_time,
                measurements[ss,comp_ind,:],
                trace_opts.meas_line,
                label=sensor_tags[ss],
                lw=plot_opts.lw,
                ms=plot_opts.ms,
                color=plot_opts.colors[ii % plot_opts.n_colors])

    #---------------------------------------------------------------------------
    # Axis / legend labels and options
    ax.set_xlabel(trace_opts.time_label,
                fontsize=plot_opts.font_ax_size, fontname=plot_opts.font_name)
    ax.set_ylabel(descriptor.create_label(comp_ind),
                fontsize=plot_opts.font_ax_size, fontname=plot_opts.font_name)

    if trace_opts.time_min_max is None:
        min_time = np.min((np.min(samp_time),np.min(sensor_time)))
        max_time = np.max((np.max(samp_time),np.max(sensor_time)))
        ax.set_xlim((min_time,max_time)) # type: ignore
    else:
        ax.set_xlim(trace_opts.time_min_max)

    if trace_opts.legend:
        ax.legend(prop={"size":plot_opts.font_leg_size},loc='best')

    plt.grid(True)
    plt.draw()

    return (fig,ax)


def plot_exp_traces(exp_sim: ExperimentSimulator,
                    component: str,
                    sens_array_num: int,
                    sim_num: int,
                    trace_opts: TraceOptsExperiment | None = None,
                    plot_opts: PlotOptsGeneral | None = None) -> tuple[Any,Any]:
    #---------------------------------------------------------------------------
    n_sensors = exp_sim.sensor_arrays[sens_array_num].sensor_data.positions.shape[0]
    descriptor = exp_sim.sensor_arrays[sens_array_num].descriptor
    comp_ind = exp_sim.sensor_arrays[sens_array_num].field.get_component_index(component)
    samp_time = exp_sim.sensor_arrays[sens_array_num].get_sample_times()
    num_sens = exp_sim.sensor_arrays[sens_array_num].get_measurement_shape()[0]

    #---------------------------------------------------------------------------
    if trace_opts is None:
        trace_opts = TraceOptsExperiment()

    if plot_opts is None:
        plot_opts = PlotOptsGeneral()

    #---------------------------------------------------------------------------
    exp_data = exp_sim.get_data()
    exp_stats = exp_sim.get_stats()

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
        for ii,ss in enumerate(sensors_to_plot):
            for ee in range(exp_sim.num_exp_per_sim):
                ax.plot(samp_time,
                        exp_data[sens_array_num][sim_num,ee,ss,comp_ind,:],
                        "+",
                        lw=plot_opts.lw,
                        ms=plot_opts.ms,
                        color=plot_opts.colors[ii % plot_opts.n_colors])

    #---------------------------------------------------------------------------
    sensor_tags = descriptor.create_sensor_tags(n_sensors)
    for ii,ss in enumerate(sensors_to_plot):
        if trace_opts.centre == "median":
            ax.plot(samp_time,
                    exp_stats[sens_array_num].median[sim_num,ss,comp_ind,:],
                    trace_opts.exp_mean_line,
                    label=sensor_tags[ss],
                    lw=plot_opts.lw,
                    ms=plot_opts.ms,
                    color=plot_opts.colors[ii % plot_opts.n_colors])
        else:
            ax.plot(samp_time,
                    exp_stats[sens_array_num].mean[sim_num,ss,comp_ind,:],
                    trace_opts.exp_mean_line,
                    label=sensor_tags[ss],
                    lw=plot_opts.lw,
                    ms=plot_opts.ms,
                    color=plot_opts.colors[ii % plot_opts.n_colors])

        if trace_opts is not None:
            upper = np.zeros_like(exp_stats[sens_array_num].min)
            lower = np.zeros_like(exp_stats[sens_array_num].min)

            if trace_opts.fill_between == 'max':
                upper = exp_stats[sens_array_num].min
                lower = exp_stats[sens_array_num].max

            elif trace_opts.fill_between == 'quartile':
                upper = exp_stats[sens_array_num].q25
                lower = exp_stats[sens_array_num].q75

            elif trace_opts.fill_between == '2std':
                upper = exp_stats[sens_array_num].mean + \
                        2*exp_stats[sens_array_num].std
                lower = exp_stats[sens_array_num].mean - \
                        2*exp_stats[sens_array_num].std

            elif trace_opts.fill_between == '3std':
                upper = exp_stats[sens_array_num].mean + \
                        3*exp_stats[sens_array_num].std
                lower = exp_stats[sens_array_num].mean - \
                        3*exp_stats[sens_array_num].std

            ax.fill_between(samp_time,
                upper[sim_num,ss,comp_ind,:],
                lower[sim_num,ss,comp_ind,:],
                color=plot_opts.colors[ii % plot_opts.n_colors],
                alpha=0.2)

    #---------------------------------------------------------------------------
    # Plot simulation and truth line
    if trace_opts.sim_line is not None:
        sim_time = exp_sim.sensor_arrays[sens_array_num].field.get_time_steps()
        sim_vals = exp_sim.sensor_arrays[sens_array_num].field.sample_field(
                    exp_sim.sensor_arrays[sens_array_num].positions)

        for ss in sensors_to_plot:
            ax.plot(sim_time,
                    sim_vals[ss,comp_ind,:],
                    trace_opts.sim_line,
                    lw=plot_opts.lw,
                    ms=plot_opts.ms)

    if trace_opts.truth_line is not None:
        truth = exp_sim.sensor_arrays[sens_array_num].get_truth()
        for ss in sensors_to_plot:
            ax.plot(samp_time,
                    truth[ss,comp_ind,:],
                    trace_opts.truth_line,
                    lw=plot_opts.lw,
                    ms=plot_opts.ms,
                    color=plot_opts.colors[ii % plot_opts.n_colors])

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

    if trace_opts.legend:
        ax.legend(prop={"size":plot_opts.font_leg_size},loc='best')

    plt.grid(True)
    plt.draw()

    return (fig,ax)
