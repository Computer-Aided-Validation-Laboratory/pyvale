# ==============================================================================
# pyvale: the python validation engine
# License: MIT
# Copyright (C) 2025 The Computer Aided Validation Team
# ==============================================================================

import math
from typing import Any
import numpy as np
import matplotlib.pyplot as plt
from pyvale.sensorsim.sensorspoint import SensorsPoint
from pyvale.sensorsim.visualopts import (PlotOptsGeneral,
                                         TraceOptsSensor)
from pyvale.sensorsim.plotting_logs import logger
from pyvale.sensorsim.logger import Logger



def subplot_calc(total_sensors: range | None,
                 sensors_per_plot: int | None) -> tuple[int,int]:
    """
    Automatically calculate the number of subplots based on
    the total number of sensors to be plot and the maximum per subplot

    Parameters
    ----------
    total_sensors : range | None
        The sensors that are to be plot
    sensors_per_plot: int | None
        The maximum amount of sensors to be plot per subplot

    Returns
    -------
    coord[int,int]
        The amount of rows and columns for the matplotlib figure.
    """

    coord = [1,1]
    sensor_num = len(total_sensors)/sensors_per_plot

    if sensor_num > 1:
        squares = math.sqrt(sensor_num)
        coordx = math.ceil(squares)
        coordy = round(squares)
        coord = [coordy, coordx]

    return coord

def make_labels(legend_loc_trace_opts,
                axs,
                leg_font_size_plot_opts,
                linestemp):

    """
    Make a legend for a filled subplot

    Parameters
    ----------
    legend_loc_trace_opts : str | None
        Legend location based on matplotlib legend location string,
        from TraceOptsSensor class.
    axs: matplotlib.axes.Axes | None
        Subplot to create legend for.
    leg_font_size_plot_opts : float | None
        Font size for legend, from PlotOptsGeneral class
    linestemp : list | None
        List of the lines plot on subplot for the legend
    """

    if legend_loc_trace_opts is not None:
        axs.legend(handles=linestemp,
        prop={"size":leg_font_size_plot_opts},
        loc=legend_loc_trace_opts, bbox_to_anchor=(1, 1))


# TODO: this should probably take an ISensorarray
def plot_time_traces(sensor_array: SensorsPoint,
                     comp_key: str | None  = None,
                     trace_opts: TraceOptsSensor | None = None,
                     plot_opts: PlotOptsGeneral | None = None
                     ) -> tuple[Any,Any]:
    """Plots time traces for the truth and virtual experiments of the sensors
    in the given sensor array.

    Parameters
    ----------
    sensor_array : SensorPoint
        The sensor array to plot times traces from.
    comp_key : str | None
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
    mylogger = Logger(__name__)
    mylogger.make_logger()
    mylogger.put_error()
    logger.info(f"Starting plotting process")
    field = sensor_array._field
    samp_time = sensor_array.get_sample_times()
    measurements = sensor_array.get_measurements()
    num_sens = sensor_array._sensor_data.positions.shape[0]
    descriptor = sensor_array._descriptor
    sensors_perturbed = (sensor_array
        .get_error_integrator()
        .get_sens_data_accumulated()
    )

    comp_ind = 0
    if comp_key is not None:
        comp_ind = sensor_array._field.get_component_index(comp_key)



    #---------------------------------------------------------------------------
    if plot_opts is None:
        plot_opts = PlotOptsGeneral()

    if trace_opts is None:
        trace_opts = TraceOptsSensor()

    if trace_opts.one_line is None:
        trace_opts.one_line = False

    if trace_opts.sensors_to_plot is None:
        if num_sens <= trace_opts.total_sensors:
            sensors_to_plot = range(num_sens)
            logger.debug(f"Only {num_sens} sensors to plot. Plotting all sensors")
        else:
            n = num_sens/trace_opts.total_sensors
            step = math.ceil(n)
            logger.info(f"Lots of sensors... Plotting sensor after every {step}... ")
            sensors_to_plot = range(0, num_sens, step)
    else:
        sensors_to_plot = trace_opts.sensors_to_plot
        sensor_list = [x+1 for x in range(num_sens)]
        fake_sensors = list(filter(lambda x: x not in sensor_list, sensors_to_plot))
        print(f"The sensors {fake_sensors} do not exist")
        for i in sensors_to_plot:
            if i not in sensor_list:
                logger.warning(f"[{i}] not a valid sensor number. Removing from sensors to plot")
                sensors_to_plot.remove(i)
        sensors_to_plot = sensors_to_plot

    if sensors_to_plot == 0:
        logger.warning("No sensors found to plot")

    if trace_opts.sensors_per_plot is None:
        sensors_per_plot = len(sensors_to_plot)+1
    elif trace_opts.sensors_per_plot > num_sens:
        logger.warning(f"Sensors per plot cannot be more than the total number of sensors. Defaulting to {num_sens} sensors per plot")
        sensors_per_plot = len(sensors_to_plot)+1
    else:
        sensors_per_plot = trace_opts.sensors_per_plot

    if sensors_per_plot > 10:
        logger.warning(f"More than 10 sensors per plot may affect plot readability, Defaulting to 10 sensors per plot...")
        sensors_per_plot = 10


    #---------------------------------------------------------------------------
    # Figure canvas setup

    if trace_opts.one_line == False:
        coords = subplot_calc(sensors_to_plot, sensors_per_plot)
    else:
        coords = (1,math.ceil(len(sensors_to_plot)/sensors_per_plot))

    fig, ax = plt.subplots(coords[0], coords[1], figsize=plot_opts.single_fig_size_landscape,
                           layout="constrained")
    fig.set_dpi(plot_opts.resolution)


    if isinstance(ax, np.ndarray) == False:
        # For a single subplot ax is a 0-dimensional np array
        # Make ax a list here so that it can be indexed as ax[0]
        # alongside 1-dimensional ax np arrays
        logger.debug(f"0-dimensional ax has been converted to a list [ax]")
        ax = [ax]
    else:
        ax = ax.flatten()

    current_plot = 0

    #---------------------------------------------------------------------------
    # Plot simulation and truth lines
    if trace_opts.sim_line is not None:
        sim_time = field.get_time_steps()
        sim_vals = field.sample_field(sensor_array._sensor_data.positions,
                                      None,
                                      sensor_array._sensor_data.angles)
        for ii,ss in enumerate(sensors_to_plot):
            if (ii+1) % sensors_per_plot == 0:
                current_plot = current_plot+1
            ax[current_plot].plot(sim_time,
                    sim_vals[ss,comp_ind,:],
                    trace_opts.sim_line,
                    lw=plot_opts.lw,
                    ms=plot_opts.ms,
                    color=plot_opts.colors[ii % plot_opts.colors_num])
        current_plot = 0

    if trace_opts.truth_line is not None:
        truth = sensor_array.get_truth()
        for ii,ss in enumerate(sensors_to_plot):
            ax[current_plot].plot(samp_time,
                    truth[ss,comp_ind,:],
                    trace_opts.truth_line,
                    lw=plot_opts.lw,
                    ms=plot_opts.ms,
                    color=plot_opts.colors[ii % plot_opts.colors_num])
            if (ii+1) % sensors_per_plot == 0:
                current_plot = current_plot+1
        current_plot = 0

    sensor_tags = descriptor.create_sensor_tags(num_sens)
    lines = []
    linestemp = []

    for ii,ss in enumerate(sensors_to_plot):
        sensor_time = samp_time

        if sensors_perturbed is not None:
            if sensors_perturbed.sample_times is not None:
                sensor_time = sensors_perturbed.sample_times
        line, = ax[current_plot].plot(sensor_time,
                measurements[ss,comp_ind,:],
                trace_opts.meas_line,
                label=sensor_tags[ss],
                lw=plot_opts.lw,
                ms=plot_opts.ms,
                color=plot_opts.colors[ii % plot_opts.colors_num])
        linestemp.append(line)
        logger.info(f"{line} has been plot")

        if (ii+1) % sensors_per_plot == 0:
            make_labels(trace_opts.legend_loc,
                        ax[current_plot],
                        plot_opts.font_leg_size,
                        linestemp)
            logger.info(f"Legend labels made for sensors: {linestemp}")
            linestemp = []

            current_plot = current_plot+1

        lines.append(line)

        if ss == sensors_to_plot[-1]:
            if (ii+1) % sensors_per_plot == 0:
                pass
            else:
                make_labels(trace_opts.legend_loc,
                            ax[current_plot],
                            plot_opts.font_leg_size,
                            linestemp)
                logger.info(f"Legend labels made for sensors: {linestemp}")

    current_plot = 0

    #---------------------------------------------------------------------------
    # Axis / legend labels and options
    ax[current_plot].set_xlabel(trace_opts.time_label,
                fontsize=plot_opts.font_ax_size, fontname=plot_opts.font_name)
    ax[current_plot].set_ylabel(descriptor.create_label(comp_ind),
                fontsize=plot_opts.font_ax_size, fontname=plot_opts.font_name)

    if trace_opts.time_min_max is None:
        min_time = np.min((np.min(samp_time),np.min(sensor_time)))
        max_time = np.max((np.max(samp_time),np.max(sensor_time)))
        ax[current_plot].set_xlim((min_time,max_time)) # type: ignore
    else:
        ax[current_plot].set_xlim(trace_opts.time_min_max)

    if trace_opts.legend_loc is not None:
        if len(ax) == 1:
            ax[0].legend(handles=lines,
                    prop={"size":plot_opts.font_leg_size},
                    loc=trace_opts.legend_loc, bbox_to_anchor=(1, 1))

    for i in ax:
        i.grid(True)
    plt.draw()

    return (fig,ax)

