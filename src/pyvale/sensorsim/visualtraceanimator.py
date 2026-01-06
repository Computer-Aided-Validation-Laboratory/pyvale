# ==============================================================================
# pyvale: the python validation engine
# License: MIT
# Copyright (C) 2025 The Computer Aided Validation Team
# ==============================================================================

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from pyvale.sensorsim.sensorspoint import SensorsPoint
from pyvale.sensorsim.visualopts import VisOptsSimSensors, VisOptsAnimation
from pyvale.sensorsim.visualopts import PlotOptsGeneral, TraceOptsSensor


def animate_trace_with_sensors(sensor_array: SensorsPoint,
                            component: str,
                            time_steps: np.ndarray | None = None,
                            trace_opts: TraceOptsSensor | None = None,
                            anim_opts: VisOptsAnimation | None = None,
                            plot_opts: PlotOptsGeneral | None = None
                            ):


    samp_time = sensor_array.get_sample_times()
    num_sens = sensor_array._sensor_data.positions.shape[0]

    if trace_opts is None:
        trace_opts = TraceOptsSensor()

    if anim_opts is None:
        anim_opts = VisOptsAnimation()

    if plot_opts is None:
        plot_opts = PlotOptsGeneral()

    if time_steps is None:
        time_steps = np.arange(0,sensor_array.get_sample_times().shape[0])

    if trace_opts.sensors_to_plot is None:
        #sensors_to_plot = trace_opts.sensors_to_plot
        sensors_to_plot = range(num_sens)
        print(sensors_to_plot)


    comp_ind = 0
    if component is not None:
        comp_ind = sensor_array._field.get_component_index(component)


    fig,ax = plt.subplots(1,1, figsize=plot_opts.single_fig_size_landscape,
                           layout="constrained")


    #ax.set_xlim([0, 10])
    #scat = ax.scatter(1, 0)
    #x = np.linspace(0, 10)


    def animate(i):
        # what should be plot...
        #scat.set_offsets((x[i], 0))
        #return (scat,)
        for tt in time_steps:
            for ii,ss in enumerate(sensors_to_plot):
                truth = sensor_array.get_truth()
                ax.plot(samp_time,
                truth[ss,comp_ind,:],
                trace_opts.truth_line,
                lw=plot_opts.lw,
                ms=plot_opts.ms,
                color=plot_opts.colors[ii % plot_opts.colors_num])

    #ani = animation.FuncAnimation(fig, animate, repeat=True, frames=len(x) - 1, interval=50)
    ani = animation.FuncAnimation(fig, animate, repeat=True, interval=50)

    plt.draw()
    plt.show()