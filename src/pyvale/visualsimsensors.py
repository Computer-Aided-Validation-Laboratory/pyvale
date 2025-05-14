# ==============================================================================
# pyvale: the python validation engine
# License: MIT
# Copyright (C) 2025 The Computer Aided Validation Team
# ==============================================================================

"""
This module contains functions for visualising virtual sensors on a simulation
mesh with simulated fields using pyvista.
"""

import vtk #NOTE: has to be here to fix latex bug in pyvista/vtk
# See: https://github.com/pyvista/pyvista/discussions/2928
#NOTE: causes output to console to be suppressed unfortunately
#NOTE: May 2025, the console suppression output is fixed but the vtk import is
#still required tro make latex work.
import pyvista as pv

import mooseherder as mh

from pyvale.sensorarraypoint import SensorArrayPoint
from pyvale.fieldconverter import simdata_to_pyvista
from pyvale.visualopts import (VisOptsSimSensors,VisOptsImageSave)
from pyvale.visualtools import (create_pv_plotter,
                                     get_colour_lims,
                                     save_pv_image)


# TODO: this needs to be updated to allow the user to plot at sensor times not
# just simulation times. This will require interpolation of the underlying
# simulation fields.
def add_sim_field(pv_plot: pv.Plotter,
                  sensor_array: SensorArrayPoint,
                  component: str,
                  time_step: int,
                  vis_opts: VisOptsSimSensors,
                  ) -> tuple[pv.Plotter,pv.UnstructuredGrid]:
    """Adds a simulation field to a pyvista plot object which is visualised on
    the mesh using a colormap.

    Parameters
    ----------
    pv_plot : pv.Plotter
        Handle to the pyvista plot object to add the simulation field to.
    sensor_array : SensorArrayPoint
        Sensor array associated with the field to be plotted.
    component : str
        String key for the field component to be shown.
    time_step : int
        Time step to plot based on the time steps in the underlying simulation
        data object.
    vis_opts : VisOptsSimSensors
        Dataclass containing options for controlling the appearance of the
        virtual sensors.

    Returns
    -------
    tuple[pv.Plotter,pv.UnstructuredGrid]
        Tuple containing a handle to the pyvista plotter which has had the field
        visualisation added and the pyvistas unstructured grid that was used to
        plot the field.
    """
    sim_vis = sensor_array._field.get_visualiser()
    sim_data = sensor_array._field.get_sim_data()
    sim_vis[component] = sim_data.node_vars[component][:,time_step]
    comp_ind = sensor_array._field.get_component_index(component)

    scalar_bar_args = {"title":sensor_array._descriptor.create_label(comp_ind),
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

    if vis_opts.time_label_pos is not None:
        pv_plot.add_text(f"Time: {sim_data.time[time_step]} " + \
                            f"{sensor_array._descriptor.time_units}",
                            position=vis_opts.time_label_pos,
                            font_size=vis_opts.time_label_font_size,
                            name='time-label')

    return (pv_plot,sim_vis)


# TODO: this should be able to take a list of ISensorArray and plot all of them
# on the same mesh.
def add_sensor_points_nom(pv_plot: pv.Plotter,
                          sensor_array: SensorArrayPoint,
                          vis_opts: VisOptsSimSensors,
                          ) -> pv.Plotter:
    """Adds points and tagged labels showing the virtual sensor locations on
    the simulation mesh in the given pyvista plot object.

    Parameters
    ----------
    pv_plot : pv.Plotter
        Pyvista plotter used to display the virtual sensor locations.
    sensor_array : SensorArrayPoint
        Sensor array for which the virtual sensor location will be shown.
    vis_opts : VisOptsSimSensors
        Dataclass containing options for controlling the appearance of the
        virtual sensors.

    Returns
    -------
    pv.Plotter
        Pyvista plotter which has had the virtual sensor locations added.
    """
    vis_sens_nominal = pv.PolyData(sensor_array._sensor_data.positions)
    vis_sens_nominal["labels"] = sensor_array._descriptor.create_sensor_tags(
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
    """Adds points showing the perturbed virtual sensor locations on
    the simulation mesh in the given pyvista plot object. Note that this will
    only work if field errors are added perturbing the sensor locations.

    Parameters
    ----------
    pv_plot : pv.Plotter
        Pyvista plotter used to display the virtual sensor locations.
    sensor_array : SensorArrayPoint
        Sensor array for which the virtual sensor location will be shown.
    vis_opts : VisOptsSimSensors
        Dataclass containing options for controlling the appearance of the
        virtual sensors.

    Returns
    -------
    pv.Plotter
        Pyvista plotter which has had the virtual sensor locations added.
    """
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
    """Plots the simulation mesh without any fields. Useful for visualising
    mesh geometry.

    Parameters
    ----------
    sim_data : mh.SimData
        Sim data object containing the mesh to plot.
    elem_dims : int
        Number of dimensions for the elements to be plotted.
    vis_opts : VisOptsSimSensors | None, optional
        Dataclass containing options for controlling the appearance of the
        virtual sensors, by default None. If None then a default options
        dataclass is created.

    Returns
    -------
    pv.Plotter
        Handle to the pyvista plotter that is showing the mesh.
    """
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
    """Plots the simulation mesh showing the specified phyiscal field at the
    time step specified.

    Parameters
    ----------
    sim_data : mh.SimData
        simulation data object containing the mesh and field data to show.
    component : str
        String key for accessing the nodal field to visualise in the sim data
        object.
    elem_dims : int
        Number of dimensions for the elements to be plotted.
    time_step : int, optional
        Simulation time step number to plot, by default -1 (the last time step).
    vis_opts : VisOptsSimSensors | None, optional
        Dataclass containing options for controlling the appearance of the
        virtual sensors, by default None. If None then a default options
        dataclass is created.

    Returns
    -------
    pv.Plotter
        Handle to the pyvista plotter showing the simulation mesh and field.
    """
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
    """Creates a visualisation of the virtual sensor locations on the simulation
    mesh showing the underlying field the sensors are sampling at the specified
    time step.

    Parameters
    ----------
    sensor_array : SensorArrayPoint
        Sensor array containing the sensors to plot and the field to display.
    component : str
        String key for accessing the nodal field to visualise in the sim data
        object.
    time_step : int, optional
        Simulation time step number to plot, by default -1 (the last time step).
    vis_opts : VisOptsSimSensors | None, optional
        Dataclass containing options for controlling the appearance of the
        virtual sensors, by default None. If None then a default options
        dataclass is created.
    image_save_opts : VisOptsImageSave | None, optional
        Dataclass containing options for saving image of the virtual sensor
        visualisation, by default None. If None a default options dataclass is
        created.

    Returns
    -------
    pv.Plotter
        Handle to the pyvista plotter showing the sensor locations.
    """
    if vis_opts is None:
        vis_opts = VisOptsSimSensors()

    sim_data = sensor_array._field.get_sim_data()
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
        save_pv_image(pv_plot,image_save_opts)

    return pv_plot

