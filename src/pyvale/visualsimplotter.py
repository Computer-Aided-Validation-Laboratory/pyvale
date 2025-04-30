# ================================================================================
# pyvale: the python validation engine
# License: MIT
# Copyright (C) 2025 The Computer Aided Validation Team
# ================================================================================

# import vtk #NOTE: has to be here to fix latex bug in pyvista/vtk
# See: https://github.com/pyvista/pyvista/discussions/2928
#NOTE: causes output to console to be suppressed unfortunately
import pyvista as pv

import mooseherder as mh

from pyvale.sensorarraypoint import SensorArrayPoint
from pyvale.fieldconverter import simdata_to_pyvista
from pyvale.visualopts import (VisOptsSimSensors,VisOptsImageSave)
from pyvale.visualtools import (create_pv_plotter,
                                     get_colour_lims,
                                     save_image)


#TODO: Docstrings

def add_sim_field(pv_plot: pv.Plotter,
                  sensor_array: SensorArrayPoint,
                  component: str,
                  time_step: int,
                  vis_opts: VisOptsSimSensors,
                  ) -> tuple[pv.Plotter,pv.UnstructuredGrid]:

    sim_vis = sensor_array.field.get_visualiser()
    sim_data = sensor_array.field.get_sim_data()
    sim_vis[component] = sim_data.node_vars[component][:,time_step]
    comp_ind = sensor_array.field.get_component_index(component)

    scalar_bar_args = {"title":sensor_array.descriptor.create_label(comp_ind),
                        "vertical":vis_opts.colour_bar_vertical,
                        "title_font_size":vis_opts.colour_bar_font_size,
                        "label_font_size":vis_opts.colour_bar_font_size}

    pv_plot.add_mesh(sim_vis,
                     scalars=component,
                     label="sim-data",
                     show_edges=vis_opts.show_edges,
                     show_scalar_bar=vis_opts.colour_bar_show,
                     scalar_bar_args=scalar_bar_args,
                     lighting=False,
                     clim=vis_opts.colour_bar_lims)

    if vis_opts.time_label_show:
        pv_plot.add_text(f"Time: {sim_data.time[time_step]} " + \
                            f"{sensor_array.descriptor.time_units}",
                            position=vis_opts.time_label_position,
                            font_size=vis_opts.time_label_font_size,
                            name='time-label')

    return (pv_plot,sim_vis)


def add_sensor_points_nom(pv_plot: pv.Plotter,
                          sensor_array: SensorArrayPoint,
                          vis_opts: VisOptsSimSensors,
                          ) -> pv.Plotter:

    vis_sens_nominal = pv.PolyData(sensor_array.sensor_data.positions)
    vis_sens_nominal["labels"] = sensor_array.descriptor.create_sensor_tags(
    sensor_array.get_measurement_shape()[0])

    # Add points to show sensor locations
    pv_plot.add_point_labels(vis_sens_nominal,"labels",
                            font_size=vis_opts.sens_label_font_size,
                            shape_color=vis_opts.sens_label_colour,
                            point_color=vis_opts.sens_colour_nom,
                            render_points_as_spheres=True,
                            point_size=vis_opts.sens_point_size,
                            always_visible=True)

    return pv_plot


def add_sensor_points_pert(pv_plot: pv.Plotter,
                           sensor_array: SensorArrayPoint,
                           vis_opts: VisOptsSimSensors,
                           ) -> pv.Plotter:

    sens_data_perturbed = sensor_array.get_sensor_data_perturbed()

    if sens_data_perturbed is not None and vis_opts.show_perturbed_pos:
        vis_sens_perturbed = pv.PolyData(sens_data_perturbed.positions)
        vis_sens_perturbed["labels"] = ["",]*sensor_array.get_measurement_shape()[0]

        pv_plot.add_point_labels(vis_sens_perturbed,"labels",
                                font_size=vis_opts.sens_label_font_size,
                                shape_color=vis_opts.sens_label_colour,
                                point_color=vis_opts.sens_colour_pert,
                                render_points_as_spheres=True,
                                point_size=vis_opts.sens_point_size,
                                always_visible=True)

    return pv_plot


def plot_sim_mesh(sim_data: mh.SimData,
                  elem_dims: int,
                  vis_opts: VisOptsSimSensors | None = None,
                  ) -> pv.Plotter:

    if vis_opts is None:
        vis_opts = VisOptsSimSensors()

    (_,sim_vis) = simdata_to_pyvista(sim_data=sim_data,
                                    components=None,
                                    elem_dims=elem_dims)

    pv_plot = create_pv_plotter(vis_opts)
    pv_plot.add_mesh(sim_vis,
                     label="sim-data",
                     show_edges=vis_opts.show_edges,
                     lighting=False)
    return pv_plot


def plot_sim_data(sim_data: mh.SimData,
                  component: str,
                  elem_dims: int,
                  time_step: int = -1,
                  vis_opts: VisOptsSimSensors | None = None
                  ) -> pv.Plotter:

    if vis_opts is None:
        vis_opts = VisOptsSimSensors()

    (_,sim_vis) = simdata_to_pyvista(sim_data,
                                              (component,),
                                              elem_dims)

    sim_vis[component] = sim_data.node_vars[component][:,time_step]

    pv_plot = create_pv_plotter(vis_opts)
    pv_plot.add_mesh(sim_vis,
                     scalars=component,
                     label="sim-data",
                     show_edges=vis_opts.show_edges,
                     show_scalar_bar=vis_opts.colour_bar_show,
                     lighting=False,
                     clim=vis_opts.colour_bar_lims)

    return pv_plot


def plot_point_sensors_on_sim(sensor_array: SensorArrayPoint,
                              component: str,
                              time_step: int = -1,
                              vis_opts: VisOptsSimSensors | None = None,
                              image_save_opts: VisOptsImageSave | None = None,
                              ) -> pv.Plotter:

    if vis_opts is None:
        vis_opts = VisOptsSimSensors()

    sim_data = sensor_array.field.get_sim_data()
    vis_opts.colour_bar_lims = get_colour_lims(
        sim_data.node_vars[component][:,time_step],
        vis_opts.colour_bar_lims)

    pv_plot = create_pv_plotter(vis_opts)

    pv_plot = add_sensor_points_pert(pv_plot,sensor_array,vis_opts)
    pv_plot = add_sensor_points_nom(pv_plot,sensor_array,vis_opts)
    (pv_plot,_) = add_sim_field(pv_plot,
                                sensor_array,
                                component,
                                time_step,
                                vis_opts)

    pv_plot.camera_position = vis_opts.camera_position

    if image_save_opts is not None:
        save_image(pv_plot,image_save_opts)

    return pv_plot

