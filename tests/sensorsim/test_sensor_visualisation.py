# ==============================================================================
# pyvale: the python validation engine
# License: MIT
# Copyright (C) 2025 The Computer Aided Validation Team
# ==============================================================================

"""Unit tests verifying 3D PyVista sensor measurement geometry builders,
universal simulation plotter, and off-screen screenshot export.
"""

from pathlib import Path
import numpy as np
import pyvista as pv

from pyvale import verif
from pyvale.sensorsim.spatialwindows import (
    SpatialWindowPoint,
    SpatialWindowLine,
    SpatialWindowRectangle,
    SpatialWindowDisk,
    SpatialWindowBox,
    SpatialWindowCylinder,
    SpatialWindowSphere,
)
from pyvale.sensorsim.visualopts import (
    VisOptsSensorGeom,
    VisOptsSimSensors,
    VisOptsImageSave,
)
from pyvale.sensorsim.visualsensormeshes import (
    build_spatial_window_mesh,
    build_sensor_array_meshes,
    build_ray_sensor_meshes,
    build_differential_sensor_meshes,
)
from pyvale.sensorsim.visualsimplotter import (
    plot_sensors_on_sim,
    add_sensor_geometries,
)
from pyvale.sensorsim.sensorlibrary import SensorLibrary
from pyvale.sensorsim.sensorsdifferential import SensorsDifferential
from pyvale.sensorsim.sensorsray import SensorsRay, ERayMode
from pyvale.sensorsim.enums import EDim


def test_vis_line_sensor_mesh() -> None:
    """Verifies that SpatialWindowLine creates an oriented 3D cylinder."""
    window = SpatialWindowLine(length=10.0)
    pos = np.array([5.0, 5.0, 0.0])
    geom_opts = VisOptsSensorGeom(line_radius=0.5)

    mesh = build_spatial_window_mesh(
        window, position=pos, geom_opts=geom_opts
    )

    assert isinstance(mesh, pv.PolyData)
    bounds = mesh.bounds  # (xmin, xmax, ymin, ymax, zmin, zmax)
    # Length 10 centered at x=5 -> bounds from 0 to 10
    assert np.isclose(bounds[0], 0.0, atol=1e-3)
    assert np.isclose(bounds[1], 10.0, atol=1e-3)


def test_vis_area_rectangle_and_circle_mesh() -> None:
    """Verifies that rectangle and circle area windows generate 3D meshes."""
    rect = SpatialWindowRectangle(length_x=8.0, length_y=4.0)
    pos = np.array([10.0, 10.0, 0.0])
    mesh_rect = build_spatial_window_mesh(rect, position=pos)

    assert isinstance(mesh_rect, pv.PolyData)
    bounds_rect = mesh_rect.bounds
    assert np.isclose(bounds_rect[1] - bounds_rect[0], 8.0, atol=1e-3)
    assert np.isclose(bounds_rect[3] - bounds_rect[2], 4.0, atol=1e-3)

    circ = SpatialWindowDisk(radius=3.0)
    mesh_circ = build_spatial_window_mesh(circ, position=pos)
    assert isinstance(mesh_circ, pv.PolyData)
    bounds_circ = mesh_circ.bounds
    assert np.isclose(bounds_circ[1] - bounds_circ[0], 6.0, atol=1e-3)
    assert np.isclose(bounds_circ[3] - bounds_circ[2], 6.0, atol=1e-3)


def test_vis_volume_meshes() -> None:
    """Verifies that box, cylinder, and sphere volume windows generate
    valid 3D meshes.
    """
    pos = np.array([0.0, 0.0, 0.0])

    box = SpatialWindowBox(length_x=4.0, length_y=6.0, length_z=8.0)
    mesh_box = build_spatial_window_mesh(box, position=pos)
    assert isinstance(mesh_box, pv.PolyData)
    b_box = mesh_box.bounds
    assert np.isclose(b_box[1] - b_box[0], 4.0, atol=1e-3)
    assert np.isclose(b_box[3] - b_box[2], 6.0, atol=1e-3)
    assert np.isclose(b_box[5] - b_box[4], 8.0, atol=1e-3)

    cyl = SpatialWindowCylinder(radius=2.0, height=5.0)
    mesh_cyl = build_spatial_window_mesh(cyl, position=pos)
    assert isinstance(mesh_cyl, pv.PolyData)

    sphere = SpatialWindowSphere(radius=3.5)
    mesh_sphere = build_spatial_window_mesh(sphere, position=pos)
    assert isinstance(mesh_sphere, pv.PolyData)
    b_sph = mesh_sphere.bounds
    assert np.isclose(b_sph[1] - b_sph[0], 7.0, atol=0.05)


def test_vis_ray_sensor_meshes() -> None:
    """Verifies that SensorsRay generates ray sightline tubes, aperture
    spheres, and target strike markers.
    """
    sim_data, _ = verif.scalar_quadratic_2d()

    ray_sensor = SensorsRay(
        sim_data=sim_data,
        ray_origins=np.array([[5.0, 3.75, 10.0]]),
        ray_directions=np.array([[0.0, 0.0, -1.0]]),
        mode=ERayMode.DISTANCE,
    )

    items = build_ray_sensor_meshes(ray_sensor)
    assert len(items) >= 2  # Tube, aperture, and strike markers
    for item in items:
        assert isinstance(item["mesh"], pv.PolyData)


def test_vis_differential_sensor_meshes() -> None:
    """Verifies that SensorsDifferential generates primary and secondary
    anchor meshes and connecting span tie rods.
    """
    sim_data, _ = verif.scalar_quadratic_2d()
    n_pts = sim_data.coords.shape[0]
    n_times = sim_data.time.shape[0]
    sim_data.node_vars["disp_x"] = np.zeros((n_pts, n_times))
    sim_data.node_vars["disp_y"] = np.zeros((n_pts, n_times))

    ext = SensorLibrary.extensometer(
        sim_data,
        anchor_a=(2.0, 3.75, 0.0),
        anchor_b=(8.0, 3.75, 0.0),
        disp_keys=("disp_x", "disp_y"),
        spatial_dims=EDim.TWOD,
    )

    items = build_differential_sensor_meshes(ext)
    assert len(items) >= 3  # Anchor A, Anchor B, Span line


def test_plot_sensors_on_sim_headless_export(tmp_path: Path) -> None:
    """Verifies that plot_sensors_on_sim runs off-screen and exports a PNG
    image to disk.
    """
    sim_data, _ = verif.scalar_quadratic_2d()
    n_pts = sim_data.coords.shape[0]
    n_times = sim_data.time.shape[0]
    sim_data.node_vars["strain_xx"] = np.zeros((n_pts, n_times))
    sim_data.node_vars["strain_yy"] = np.zeros((n_pts, n_times))
    sim_data.node_vars["strain_xy"] = np.zeros((n_pts, n_times))

    gauge = SensorLibrary.strain_gauge(
        sim_data,
        positions=np.array([[5.0, 3.75, 0.0]]),
        grid_length_x=3.0,
        grid_length_y=1.5,
        spatial_dims=EDim.TWOD,
    )

    out_file = tmp_path / "test_vis_render.png"
    vis_opts = VisOptsSimSensors()
    geom_opts = VisOptsSensorGeom(area_opacity=0.8)
    save_opts = VisOptsImageSave(path=out_file)

    # Call plot_sensors_on_sim
    pv_plot = plot_sensors_on_sim(
        sensor_array=gauge,
        component="strain_xx",
        vis_opts=vis_opts,
        geom_opts=geom_opts,
        image_save_opts=save_opts,
    )
    assert pv_plot is not None
    pv_plot.close()

    assert out_file.exists()
    assert out_file.stat().st_size > 0
