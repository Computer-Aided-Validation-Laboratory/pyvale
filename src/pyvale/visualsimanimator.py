# ================================================================================
# pyvale: the python validation engine
# License: MIT
# Copyright (C) 2025 The Computer Aided Validation Team
# ================================================================================

import numpy as np
#import vtk #NOTE: has to be here to fix latex bug in pyvista/vtk
# See: https://github.com/pyvista/pyvista/discussions/2928
#NOTE: causes output to console to be suppressed unfortunately
import pyvista as pv
from pyvale.sensorarraypoint import SensorArrayPoint
from pyvale.visualopts import VisOptsSimSensors, VisOptsAnimation
from pyvale.visualtools import (create_pv_plotter,
                                     get_colour_lims,
                                     set_animation_writer)
from pyvale.visualsimplotter import (add_sensor_points_nom,
                                          add_sensor_points_pert,
                                          add_sim_field)

#TODO: Docstrings

def animate_sim_with_sensors(sensor_array: SensorArrayPoint,
                            component: str,
                            time_steps: np.ndarray | None = None,
                            vis_opts: VisOptsSimSensors | None = None,
                            anim_opts: VisOptsAnimation | None = None,
                            ) -> pv.Plotter:

    if vis_opts is None:
        vis_opts = VisOptsSimSensors()

    if anim_opts is None:
        anim_opts = VisOptsAnimation()

    if time_steps is None:
        time_steps = np.arange(0,sensor_array.get_sample_times().shape[0])

    sim_data = sensor_array.field.get_sim_data()
    vis_opts.colour_bar_lims = get_colour_lims(
        sim_data.node_vars[component][:,time_steps],
        vis_opts.colour_bar_lims)

    #---------------------------------------------------------------------------
    pv_plot = create_pv_plotter(vis_opts)

    pv_plot = add_sensor_points_pert(pv_plot,sensor_array,vis_opts)
    pv_plot = add_sensor_points_nom(pv_plot,sensor_array,vis_opts)
    (pv_plot,sim_vis) = add_sim_field(pv_plot,
                                      sensor_array,
                                      component,
                                      time_step = 0,
                                      vis_opts = vis_opts)

    pv_plot.camera_position = vis_opts.camera_position
    pv_plot.show(auto_close=False,interactive=False)

    pv_plot = set_animation_writer(pv_plot,anim_opts)

    #---------------------------------------------------------------------------
    for tt in time_steps:
        # Updates the field plotted on the mesh
        sim_vis[component] = sim_data.node_vars[component][:,tt]

        if vis_opts.time_label_show:
            pv_plot.add_text(f"Time: {sim_data.time[tt]} " + \
                             f"{sensor_array.descriptor.time_units}",
                             position=vis_opts.time_label_position,
                             font_size=vis_opts.time_label_font_size,
                             name='time-label')

        if anim_opts.save_animation is not None:
            pv_plot.write_frame()

    pv_plot.show(auto_close=False,interactive=vis_opts.interactive)
    return pv_plot






