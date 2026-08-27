#===============================================================================
# pyvale: the python validation engine
# License: MIT
# Copyright (C) 2025 The Computer Aided Validation Team
#===============================================================================
# import vtk #NOTE: has to be here to fix latex bug in pyvista/vtk
# See: https://github.com/pyvista/pyvista/discussions/2928
#NOTE: causes output to console to be suppressed unfortunately
from pathlib import Path
import numpy as np
import pyvista as pv

import pyvale.mooseherder as mh
import pyvale.dataio as io

from pyvale.sensorsim.sensorspoint import SensorsPoint
from pyvale.sensorsim.sensorarray import ISensorArray
from pyvale.sensorsim.fieldconverter import simdata_to_pyvista_vis
from pyvale.sensorsim.visualopts import (
    VisOptsSimSensors,
    VisOptsImageSave,
    VisOptsSensorGeom,
)
from pyvale.sensorsim.visualsensormeshes import build_sensor_array_meshes
from pyvale.sensorsim.visualtools import (
    create_pv_plotter,
    get_colour_lims,
)
#TODO: Docstrings

def add_sim_field(pv_plot: pv.Plotter,
                  sensor_array: SensorsPoint,
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
                          sensor_array: SensorsPoint,
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
                           sensor_array: SensorsPoint,
                           vis_opts: VisOptsSimSensors,
                           ) -> pv.Plotter:

    sens_data_perturbed = sensor_array.get_sensor_data_perturbed()

    if sens_data_perturbed is not None and vis_opts.show_perturbed_pos:
        vis_sens_perturbed = pv.PolyData(sens_data_perturbed.positions)
        n_sens = sensor_array.get_measurement_shape()[0]
        vis_sens_perturbed["labels"] = [""] * n_sens

        pv_plot.add_point_labels(vis_sens_perturbed,"labels",
                                font_size=vis_opts.sens_label_font_size,
                                shape_color=vis_opts.sens_label_colour,
                                point_color=vis_opts.sens_colour_pert,
                                render_points_as_spheres=True,
                                point_size=vis_opts.sens_point_size,
                                always_visible=True)

    return pv_plot


def plot_sim_mesh(sim_data: io.SimData,
                  vis_opts: VisOptsSimSensors | None = None,
                  ) -> pv.Plotter:

    if vis_opts is None:
        vis_opts = VisOptsSimSensors()

    pv_simdata = simdata_to_pyvista_vis(sim_data,
                                         None,
                                         sim_data.num_spat_dims)

    pv_plot = create_pv_plotter(vis_opts)

    pv_plot.add_mesh(pv_simdata,
                     label='sim-data',
                     show_edges=True,
                     show_scalar_bar=False)

    return pv_plot


def plot_sim_data(sim_data: io.SimData,
                  component: str,
                  time_step: int = -1,
                  vis_opts: VisOptsSimSensors | None = None
                  ) -> pv.Plotter:

    if vis_opts is None:
        vis_opts = VisOptsSimSensors()

    pv_simdata = simdata_to_pyvista_vis(sim_data,
                                        (component,),
                                         sim_data.num_spat_dims)

    pv_plot = create_pv_plotter(vis_opts)

    pv_plot.add_mesh(pv_simdata,
                     scalars=pv_simdata[component][:,time_step],
                     label="sim-data",
                     show_edges=True,
                     show_scalar_bar=True,
                     scalar_bar_args={"title":component},)


    return pv_plot


def plot_point_sensors_on_sim(sensor_array: SensorsPoint,
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

    if image_save_opts is not None and image_save_opts.path is not None:
        pv_plot.screenshot(
            str(image_save_opts.path),
            transparent_background=image_save_opts.transparent_background,
        )

    return pv_plot


def add_sensor_geometries(
    pv_plot: pv.Plotter,
    sensor_array: ISensorArray,
    geom_opts: VisOptsSensorGeom | None = None,
    time_step: int = -1,
) -> pv.Plotter:
    """Adds 3D sensor visual geometry meshes (cylinders, rectangles, discs,
    volume shells, ray tubes, view cones) to a PyVista Plotter.
    """
    if geom_opts is None:
        geom_opts = VisOptsSensorGeom()

    mesh_items = build_sensor_array_meshes(sensor_array, geom_opts=geom_opts)

    for item in mesh_items:
        pv_plot.add_mesh(
            item["mesh"],
            color=item["color"],
            opacity=item["opacity"],
            show_edges=item["show_edges"],
        )

    return pv_plot


def plot_sensors_on_sim(
    sensor_array: ISensorArray,
    component: str | None = None,
    time_step: int = -1,
    vis_opts: VisOptsSimSensors | None = None,
    geom_opts: VisOptsSensorGeom | None = None,
    image_save_opts: VisOptsImageSave | None = None,
) -> pv.Plotter:
    """Universal 3D plotter rendering simulation fields and 3D sensor
    measurement support geometries (points, lines, areas, volumes, rays,
    and differential assemblies).

    Parameters
    ----------
    sensor_array : ISensorArray
        Sensor array to visualize.
    component : str | None, optional
        Simulation field component to display as background scalar contour.
        If None, the simulation mesh is displayed without scalar contouring.
    time_step : int, optional
        Time step index to render. Defaults to last step (-1).
    vis_opts : VisOptsSimSensors | None, optional
        Plotter styling and scalar bar options.
    geom_opts : VisOptsSensorGeom | None, optional
        3D sensor geometry rendering options.
    image_save_opts : VisOptsImageSave | None, optional
        Off-screen image saving options.

    Returns
    -------
    pv.Plotter
        Configured PyVista plotter instance.
    """
    if vis_opts is None:
        vis_opts = VisOptsSimSensors()

    if geom_opts is None:
        geom_opts = VisOptsSensorGeom()

    off_scr = True if image_save_opts is not None else None
    pv_plot = create_pv_plotter(vis_opts, off_screen=off_scr)

    # 1. Add background simulation mesh/contour
    field = getattr(sensor_array, "get_field", None)
    if field is not None:
        field_obj = field()
    elif hasattr(sensor_array, "get_sensor_a"):
        field_obj = sensor_array.get_sensor_a().get_field()
    else:
        field_obj = None

    if field_obj is not None:
        sim_vis = field_obj.get_visualiser()
        sim_data = field_obj.get_sim_data()

        if component is not None and component in sim_data.node_vars:
            comp_vals = sim_data.node_vars[component][:, time_step]
            sim_vis[component] = comp_vals
            comp_ind = field_obj.get_component_index(component)
            desc = sensor_array.get_descriptor()
            title_str = (
                desc.create_label(comp_ind) if desc is not None else component
            )

            scalar_bar_args = {
                "title": title_str,
                "vertical": vis_opts.colour_bar_vertical,
                "title_font_size": vis_opts.colour_bar_font_size,
                "label_font_size": vis_opts.colour_bar_font_size,
            }

            c_min = float(np.min(comp_vals))
            c_max = float(np.max(comp_vals))
            if np.isclose(c_min, c_max):
                vis_opts.colour_bar_lims = (c_min - 1.0, c_max + 1.0)
            else:
                vis_opts.colour_bar_lims = get_colour_lims(
                    comp_vals,
                    vis_opts.colour_bar_lims,
                )

            pv_plot.add_mesh(
                sim_vis,
                scalars=component,
                label="sim-data",
                show_edges=vis_opts.show_edges,
                show_scalar_bar=vis_opts.colour_bar_show,
                scalar_bar_args=scalar_bar_args,
                lighting=False,
                clim=vis_opts.colour_bar_lims,
            )

            if vis_opts.time_label_pos is not None:
                t_val = sim_data.time[time_step]
                t_unit = desc.time_units if desc is not None else "s"
                pv_plot.add_text(
                    f"Time: {t_val} {t_unit}",
                    position=vis_opts.time_label_pos,
                    font_size=vis_opts.time_label_font_size,
                    name="time-label",
                )
        else:
            pv_plot.add_mesh(
                sim_vis,
                label="sim-data",
                show_edges=vis_opts.show_edges,
                show_scalar_bar=False,
                color="lightgray",
            )

    # 2. Add 3D sensor geometries
    add_sensor_geometries(
        pv_plot,
        sensor_array,
        geom_opts=geom_opts,
        time_step=time_step,
    )

    pv_plot.camera_position = vis_opts.camera_position

    # 3. Off-screen screenshot save if requested
    if image_save_opts is not None and image_save_opts.path is not None:
        save_p = Path(image_save_opts.path)
        save_p.parent.mkdir(parents=True, exist_ok=True)
        pv_plot.screenshot(
            str(save_p),
            transparent_background=image_save_opts.transparent_background,
        )

    return pv_plot

