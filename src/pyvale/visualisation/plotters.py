'''
================================================================================
pyvale: the python validation engine
License: MIT
Copyright (C) 2024 The Digital Validation Team
================================================================================
'''
from typing import Any

import numpy as np
import matplotlib.pyplot as plt
import vtk #NOTE: has to be here to fix latex bug in pyvista/vtk
# See: https://github.com/pyvista/pyvista/discussions/2928
#NOTE: causes output to console to be suppressed unfortunately
import pyvista as pv

import mooseherder as mh

from pyvale.physics.field import conv_simdata_to_pyvista
from pyvale.sensors.pointsensorarray import PointSensorArray
from pyvale.visualisation.plotopts import (GeneralPlotOpts,
                                           SensorTraceOpts,
                                           ExpTraceOpts)
from pyvale.experimentsimulator import ExperimentSimulator


def plot_sim_mesh(sim_data: mh.SimData) -> Any:
    pv_simdata = conv_simdata_to_pyvista(sim_data,
                                         None,
                                         sim_data.num_spat_dims)

    pv_plot = pv.Plotter(window_size=[1280, 800]) # type: ignore

    pv_plot.add_mesh(pv_simdata,
                     label='sim-data',
                     show_edges=True,
                     show_scalar_bar=False)

    pv_plot.add_axes_at_origin(labels_off=True)
    return pv_plot

def plot_sim_data(sim_data: mh.SimData,
                  component: str,
                  time_step: int = -1) -> Any:
    pv_simdata = conv_simdata_to_pyvista(sim_data,
                                        (component,),
                                         sim_data.num_spat_dims)

    pv_plot = pv.Plotter(window_size=[1280, 800]) # type: ignore

    pv_plot.add_mesh(pv_simdata,
                     scalars=pv_simdata[component][:,time_step],
                     label='sim-data',
                     show_edges=True,
                     show_scalar_bar=False)

    pv_plot.add_scalar_bar(component)
    pv_plot.add_axes_at_origin(labels_off=True)
    return pv_plot


def plot_sensors_on_sim(sensor_array: PointSensorArray,
                        component: str,
                        time_step: int = -1,
                        ) -> Any:

    pv_simdata = sensor_array.field.get_visualiser()
    pv_sensdata = sensor_array.get_visualiser()
    comp_ind = sensor_array.field.get_component_index(component)

    descriptor = sensor_array.descriptor
    pv_sensdata['labels'] = descriptor.create_sensor_tags(
        sensor_array.get_measurement_shape()[0])

    pv_plot = pv.Plotter(window_size=[1280, 800]) # type: ignore

    pv_plot.add_point_labels(pv_sensdata, "labels",
                            font_size=40,
                            shape_color='grey',
                            point_color='red',
                            render_points_as_spheres=True,
                            point_size=20,
                            always_visible=True
                            )

    pv_plot.add_mesh(pv_simdata,
                     scalars=pv_simdata[component][:,time_step],
                     label='sim-data',
                     show_edges=True,
                     show_scalar_bar=False)

    pv_plot.add_scalar_bar(descriptor.create_label(comp_ind),
                           vertical=True)
    pv_plot.add_axes_at_origin(labels_off=True)

    return pv_plot


def plot_time_traces(sensor_array: PointSensorArray,
                     component: str,
                     trace_opts: SensorTraceOpts | None = None,
                     plot_opts: GeneralPlotOpts | None = None
                     ) -> tuple[Any,Any]:

    field = sensor_array.field
    comp_ind = sensor_array.field.get_component_index(component)
    samp_time = sensor_array.get_sample_times()
    measurements = sensor_array.get_measurements()
    n_sensors = sensor_array.positions.shape[0]
    descriptor = sensor_array.descriptor

    if plot_opts is None:
        plot_opts = GeneralPlotOpts()

    if trace_opts is None:
        trace_opts = SensorTraceOpts()

    if trace_opts.sensors_to_plot is None:
        trace_opts.sensors_to_plot = np.arange(0,n_sensors)

    #---------------------------------------------------------------------------
    # Figure canvas setup
    fig, ax = plt.subplots(figsize=plot_opts.single_fig_size,
                           layout='constrained')
    fig.set_dpi(plot_opts.resolution)

    #---------------------------------------------------------------------------
    # Plot simulation and truth lines
    if trace_opts.sim_line is not None:
        sim_time = field.get_time_steps()
        sim_vals = field.sample_field(sensor_array.positions)

        for ss in range(n_sensors):
            if ss in trace_opts.sensors_to_plot:
                ax.plot(sim_time,
                        sim_vals[ss,comp_ind,:],
                        trace_opts.sim_line,
                        lw=plot_opts.lw,
                        ms=plot_opts.ms,
                        color=plot_opts.colors[ss % plot_opts.n_colors])

    if trace_opts.truth_line is not None:
        truth = sensor_array.get_truth_values()
        for ss in range(n_sensors):
            if ss in trace_opts.sensors_to_plot:
                ax.plot(samp_time,
                        truth[ss,comp_ind,:],
                        trace_opts.truth_line,
                        lw=plot_opts.lw,
                        ms=plot_opts.ms,
                        color=plot_opts.colors[ss % plot_opts.n_colors])

    sensor_tags = descriptor.create_sensor_tags(n_sensors)
    for ss in range(n_sensors):
        if ss in trace_opts.sensors_to_plot:
            ax.plot(samp_time,
                    measurements[ss,comp_ind,:],
                    trace_opts.meas_line,
                    label=sensor_tags[ss],
                    lw=plot_opts.lw,
                    ms=plot_opts.ms,
                    color=plot_opts.colors[ss % plot_opts.n_colors])

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


def plot_exp_traces(exp_sim: ExperimentSimulator,
                    component: str,
                    sens_array_num: int,
                    sim_num: int,
                    trace_opts: ExpTraceOpts | None = None,
                    plot_opts: GeneralPlotOpts | None = None) -> tuple[Any,Any]:

    if trace_opts is None:
        trace_opts = ExpTraceOpts()

    if plot_opts is None:
        plot_opts = GeneralPlotOpts()

    descriptor = exp_sim.sensor_arrays[sens_array_num].descriptor
    comp_ind = exp_sim.sensor_arrays[sens_array_num].field.get_component_index(component)
    samp_time = exp_sim.sensor_arrays[sens_array_num].get_sample_times()
    num_sens = exp_sim.sensor_arrays[sens_array_num].get_measurement_shape()[0]

    exp_data = exp_sim.get_data()
    exp_stats = exp_sim.get_stats()

    if trace_opts.sensors_to_plot is None:
        sensors_to_plot = range(num_sens)

    #---------------------------------------------------------------------------
    # Figure canvas setup
    fig, ax = plt.subplots(figsize=plot_opts.single_fig_size,
                           layout='constrained')
    fig.set_dpi(plot_opts.resolution)

    #---------------------------------------------------------------------------
    # Plot all simulated experimental points
    if trace_opts.plot_all_exp_points:
        for ss in sensors_to_plot:
            for ee in range(exp_sim.num_exp_per_sim):
                ax.plot(samp_time,
                        exp_data[sens_array_num][sim_num,ee,ss,comp_ind,:],
                        "+",
                        lw=plot_opts.lw,
                        ms=plot_opts.ms,
                        color=plot_opts.colors[ss % plot_opts.n_colors])

    for ss in sensors_to_plot:
        if trace_opts.centre == "median":
            ax.plot(samp_time,
                    exp_stats[sens_array_num].median[sim_num,ss,comp_ind,:],
                    trace_opts.exp_mean_line,
                    lw=plot_opts.lw,
                    ms=plot_opts.ms,
                    color=plot_opts.colors[ss % plot_opts.n_colors])
        else:
            ax.plot(samp_time,
                    exp_stats[sens_array_num].mean[sim_num,ss,comp_ind,:],
                    trace_opts.exp_mean_line,
                    lw=plot_opts.lw,
                    ms=plot_opts.ms,
                    color=plot_opts.colors[ss % plot_opts.n_colors])

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
                color=plot_opts.colors[ss % plot_opts.n_colors],
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
        truth = exp_sim.sensor_arrays[sens_array_num].get_truth_values()
        for ss in sensors_to_plot:
            ax.plot(samp_time,
                    truth[ss,comp_ind,:],
                    trace_opts.truth_line,
                    lw=plot_opts.lw,
                    ms=plot_opts.ms,
                    color=plot_opts.colors[ss % plot_opts.n_colors])

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

    trace_opts.legend = False
    if trace_opts.legend:
        ax.legend(prop={"size":plot_opts.font_leg_size},loc='best')

    plt.grid(True)
    plt.draw()

    return (fig,ax)