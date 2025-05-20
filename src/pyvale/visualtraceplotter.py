# ==============================================================================
# pyvale: the python validation engine
# License: MIT
# Copyright (C) 2025 The Computer Aided Validation Team
# ==============================================================================

from typing import Any
import numpy as np
import matplotlib.pyplot as plt
from pyvale.sensorarraypoint import SensorArrayPoint
from pyvale.visualopts import (PlotOptsGeneral,
                               TraceOptsSensor)



# TODO: this should probably take an ISensorarray
def plot_time_traces(sensor_array: SensorArrayPoint,
                     component: str | None  = None,
                     trace_opts: TraceOptsSensor | None = None,
                     plot_opts: PlotOptsGeneral | None = None
                     ) -> tuple[Any,Any]:
    """Plots time traces for the truth and virtual experiments of the sensors
    in the given sensor array.

    Parameters
    ----------
    sensor_array : SensorArrayPoint
        _description
    component : str | None
        String key for the field component to plot, by default None. If None
        then the first component in the measurement array is plotted
    trace_opts : TraceOptsSensor | None, optional
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
    """
    #---------------------------------------------------------------------------
    field = sensor_array._field
    samp_time = sensor_array.get_sample_times()
    measurements = sensor_array.get_measurements()
    num_sens = sensor_array._sensor_data.positions.shape[0]
    descriptor = sensor_array._descriptor
    sensors_perturbed = sensor_array.get_sensor_data_perturbed()

    comp_ind = 0
    if component is not None:
        comp_ind = sensor_array._field.get_component_index(component)

    #---------------------------------------------------------------------------
    if plot_opts is None:
        plot_opts = PlotOptsGeneral()

    if trace_opts is None:
        trace_opts = TraceOptsSensor()

    if trace_opts.sensors_to_plot is None:
        sensors_to_plot = range(num_sens)
    else:
        sensors_to_plot = trace_opts.sensors_to_plot

    #---------------------------------------------------------------------------
    # Figure canvas setup
    fig, ax = plt.subplots(figsize=plot_opts.single_fig_size_landscape,
                           layout="constrained")
    fig.set_dpi(plot_opts.resolution)

    #---------------------------------------------------------------------------
    # Plot simulation and truth lines
    if trace_opts.sim_line is not None:
        sim_time = field.get_time_steps()
        sim_vals = field.sample_field(sensor_array._sensor_data.positions,
                                      None,
                                      sensor_array._sensor_data.angles)

        for ii,ss in enumerate(sensors_to_plot):
            ax.plot(sim_time,
                    sim_vals[ss,comp_ind,:],
                    trace_opts.sim_line,
                    lw=plot_opts.lw,
                    ms=plot_opts.ms,
                    color=plot_opts.colors[ii % plot_opts.colors_num])

    if trace_opts.truth_line is not None:
        truth = sensor_array.get_truth()
        for ii,ss in enumerate(sensors_to_plot):
            ax.plot(samp_time,
                    truth[ss,comp_ind,:],
                    trace_opts.truth_line,
                    lw=plot_opts.lw,
                    ms=plot_opts.ms,
                    color=plot_opts.colors[ii % plot_opts.colors_num])

    sensor_tags = descriptor.create_sensor_tags(num_sens)
    lines = []
    for ii,ss in enumerate(sensors_to_plot):
        sensor_time = samp_time
        if sensors_perturbed is not None:
            if sensors_perturbed.sample_times is not None:
                sensor_time = sensors_perturbed.sample_times

        line, = ax.plot(sensor_time,
                measurements[ss,comp_ind,:],
                trace_opts.meas_line,
                label=sensor_tags[ss],
                lw=plot_opts.lw,
                ms=plot_opts.ms,
                color=plot_opts.colors[ii % plot_opts.colors_num])

        lines.append(line)

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

    if trace_opts.legend_loc is not None:
        ax.legend(handles=lines,
                  prop={"size":plot_opts.font_leg_size},
                  loc=trace_opts.legend_loc)

    plt.grid(True)
    plt.draw()

    return (fig,ax)

