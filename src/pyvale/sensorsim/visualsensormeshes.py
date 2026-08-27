# ==============================================================================
# pyvale: the python validation engine
# License: MIT
# Copyright (C) 2025 The Computer Aided Validation Team
# ==============================================================================

"""3D PyVista visual geometry builders for physical sensor support windows,
ray-casting sightlines, view cones, and differential multi-anchor assemblies.
"""

from typing import Any
import numpy as np
import pyvista as pv

from pyvale.sensorsim.spatialwindows import (
    ISpatialWindow,
    SpatialWindowPoint,
    SpatialWindowLine,
    SpatialWindowRectangle,
    SpatialWindowDisk,
    SpatialWindowBox,
    SpatialWindowCylinder,
    SpatialWindowSphere,
)
from pyvale.sensorsim.visualopts import VisOptsSensorGeom


def build_spatial_window_mesh(
    window: ISpatialWindow,
    position: np.ndarray,
    rotation_matrix: np.ndarray | None = None,
    geom_opts: VisOptsSensorGeom | None = None,
) -> pv.PolyData | pv.UnstructuredGrid:
    """Constructs a 3D PyVista geometric mesh representing a single spatial
    sensor measurement window positioned and oriented in 3D space.

    Parameters
    ----------
    window : ISpatialWindow
        Spatial support geometry (point, line, rectangle, circle, box, etc.).
    position : np.ndarray
        3D spatial coordinates [x, y, z] of the sensor center.
    rotation_matrix : np.ndarray | None, optional
        3x3 rotation matrix mapping local sensor coordinates to global space.
    geom_opts : VisOptsSensorGeom | None, optional
        Visual styling options (tube radius, etc.).

    Returns
    -------
    pv.PolyData | pv.UnstructuredGrid
        Transformed 3D mesh ready for rendering in a PyVista Plotter.
    """
    if geom_opts is None:
        geom_opts = VisOptsSensorGeom()

    r_mat = rotation_matrix if rotation_matrix is not None else np.eye(3)
    pos = np.asarray(position, dtype=float).flatten()[:3]

    if isinstance(window, SpatialWindowPoint):
        # Render point as a small sphere
        sphere = pv.Sphere(radius=geom_opts.line_radius, center=pos)
        return sphere

    elif isinstance(window, SpatialWindowLine):
        length = float(window.get_length())
        radius = float(geom_opts.line_radius)
        # Local line along x-axis from -L/2 to +L/2
        cyl = pv.Cylinder(
            center=(0.0, 0.0, 0.0),
            direction=(1.0, 0.0, 0.0),
            radius=radius,
            height=length,
            resolution=24,
        )
        return _transform_mesh(cyl, pos, r_mat)

    elif isinstance(window, SpatialWindowRectangle):
        w = float(window.get_length_x())
        h = float(window.get_length_y())
        # Plane in xy-plane with normal (0, 0, 1)
        plane = pv.Plane(
            center=(0.0, 0.0, 0.0),
            direction=(0.0, 0.0, 1.0),
            i_size=w,
            j_size=h,
            i_resolution=1,
            j_resolution=1,
        )
        return _transform_mesh(plane, pos, r_mat)

    elif isinstance(window, SpatialWindowDisk):
        r = float(window.get_radius())
        disc = pv.Disc(
            center=(0.0, 0.0, 0.0),
            inner=0.0,
            outer=r,
            normal=(0.0, 0.0, 1.0),
            r_res=1,
            c_res=32,
        )
        return _transform_mesh(disc, pos, r_mat)

    elif isinstance(window, SpatialWindowBox):
        lx = float(window.get_length_x())
        ly = float(window.get_length_y())
        lz = float(window.get_length_z())
        cube = pv.Cube(
            center=(0.0, 0.0, 0.0),
            x_length=lx,
            y_length=ly,
            z_length=lz,
        )
        return _transform_mesh(cube, pos, r_mat)

    elif isinstance(window, SpatialWindowCylinder):
        r = float(window.get_radius())
        h = float(window.get_height())
        cyl = pv.Cylinder(
            center=(0.0, 0.0, 0.0),
            direction=(0.0, 0.0, 1.0),
            radius=r,
            height=h,
            resolution=24,
        )
        return _transform_mesh(cyl, pos, r_mat)

    elif isinstance(window, SpatialWindowSphere):
        r = float(window.get_radius())
        sphere = pv.Sphere(
            radius=r,
            center=pos,
            theta_resolution=24,
            phi_resolution=24,
        )
        return sphere

    else:
        # Fallback to sphere
        return pv.Sphere(radius=geom_opts.line_radius, center=pos)


def build_sensor_array_meshes(
    sensor_array: Any,
    geom_opts: VisOptsSensorGeom | None = None,
) -> list[dict[str, Any]]:
    """Builds rendering mesh items for all sensors in an ISensorArray.

    Returns a list of dicts:
    [{'mesh': pv.PolyData, 'color': str, 'opacity': float, 'show_edges': bool}]
    """
    if geom_opts is None:
        geom_opts = VisOptsSensorGeom()

    items: list[dict[str, Any]] = []

    # Check for SensorsRay
    if hasattr(sensor_array, "get_ray_origins") and hasattr(
        sensor_array, "calc_ray_intersections"
    ):
        return build_ray_sensor_meshes(sensor_array, geom_opts=geom_opts)

    # Check for SensorsDifferential
    if hasattr(sensor_array, "get_sensor_a"):
        return build_differential_sensor_meshes(
            sensor_array, geom_opts=geom_opts
        )

    # Standard Spatial / Point sensor array
    sens_data = sensor_array.get_sensor_data()
    n_sens = sens_data.positions.shape[0]

    window = getattr(sensor_array, "get_spatial_window", lambda: None)()
    if window is None:
        window = SpatialWindowPoint()

    is_volume = isinstance(
        window, (SpatialWindowBox, SpatialWindowCylinder, SpatialWindowSphere)
    )
    is_area = isinstance(window, (SpatialWindowRectangle, SpatialWindowDisk))

    opacity = (
        geom_opts.volume_opacity
        if is_volume
        else (geom_opts.area_opacity if is_area else 1.0)
    )

    for ii in range(n_sens):
        pos = sens_data.positions[ii]
        if sens_data.angles is not None:
            if len(sens_data.angles) > ii:
                r_mat = sens_data.angles[ii].as_matrix()
            elif len(sens_data.angles) > 0:
                r_mat = sens_data.angles[0].as_matrix()
            else:
                r_mat = None
        else:
            r_mat = None

        mesh = build_spatial_window_mesh(window, pos, r_mat, geom_opts)
        show_edges = (
            geom_opts.show_wireframe_edges and (is_area or is_volume)
        )
        items.append({
            "mesh": mesh,
            "color": geom_opts.color_nominal,
            "opacity": opacity,
            "show_edges": show_edges,
        })

    return items


def build_ray_sensor_meshes(
    ray_sensor: Any,
    time_step: int = -1,
    geom_opts: VisOptsSensorGeom | None = None,
) -> list[dict[str, Any]]:
    """Builds 3D beam tubes, strike points, and view cones for SensorsRay."""
    if geom_opts is None:
        geom_opts = VisOptsSensorGeom()

    items: list[dict[str, Any]] = []

    origins = ray_sensor.get_ray_origins()
    dirs = ray_sensor.get_ray_directions()
    n_rays = origins.shape[0]

    hits, dists, valid = ray_sensor.calc_ray_intersections(time_step=time_step)

    radius = geom_opts.ray_tube_radius

    for ii in range(n_rays):
        p0 = origins[ii]
        is_hit = bool(valid[ii])
        p1 = hits[ii] if is_hit else (p0 + dirs[ii] * 50.0)

        # Ray tube line from p0 to p1
        line = pv.Line(p0, p1)
        tube = line.tube(radius=radius, n_sides=16)

        color = geom_opts.color_nominal if is_hit else "gray"
        items.append({
            "mesh": tube,
            "color": color,
            "opacity": 0.85,
            "show_edges": False,
        })

        # Aperture sphere
        aperture = pv.Sphere(radius=radius * 2.0, center=p0)
        items.append({
            "mesh": aperture,
            "color": "blue",
            "opacity": 1.0,
            "show_edges": False,
        })

        if is_hit:
            # Surface strike marker
            strike = pv.Sphere(radius=radius * 1.5, center=p1)
            items.append({
                "mesh": strike,
                "color": "yellow",
                "opacity": 1.0,
                "show_edges": False,
            })

            # View cone from p0 to p1
            length = float(np.linalg.norm(p1 - p0))
            if length > 1e-3:
                cone = pv.Cone(
                    center=p0 + dirs[ii] * (length * 0.5),
                    direction=dirs[ii],
                    height=length,
                    radius=length * np.tan(np.radians(5.0)),
                    resolution=24,
                )
                items.append({
                    "mesh": cone,
                    "color": "cyan",
                    "opacity": geom_opts.ray_cone_opacity,
                    "show_edges": False,
                })

    return items


def build_differential_sensor_meshes(
    diff_sensor: Any,
    geom_opts: VisOptsSensorGeom | None = None,
) -> list[dict[str, Any]]:
    """Builds dual-anchor meshes and connecting gauge span tie rods for
    SensorsDifferential.
    """
    if geom_opts is None:
        geom_opts = VisOptsSensorGeom()

    items: list[dict[str, Any]] = []

    sens_a = diff_sensor.get_sensor_a()
    sens_b = diff_sensor.get_sensor_b()

    items_a = build_sensor_array_meshes(sens_a, geom_opts=geom_opts)
    for it in items_a:
        it["color"] = "blue"
    items.extend(items_a)

    items_b = build_sensor_array_meshes(sens_b, geom_opts=geom_opts)
    for it in items_b:
        it["color"] = "red"
    items.extend(items_b)

    # Connecting gauge span tie lines
    pos_a = sens_a.get_sensor_data().positions
    pos_b = sens_b.get_sensor_data().positions
    n_pairs = min(pos_a.shape[0], pos_b.shape[0])

    for ii in range(n_pairs):
        pa = pos_a[ii]
        pb = pos_b[ii]
        line = pv.Line(pa, pb)
        tube = line.tube(radius=geom_opts.line_radius * 0.4, n_sides=12)
        items.append({
            "mesh": tube,
            "color": "black",
            "opacity": 0.75,
            "show_edges": False,
        })

    return items


def _transform_mesh(
    mesh: pv.PolyData,
    position: np.ndarray,
    rotation_matrix: np.ndarray,
) -> pv.PolyData:
    """Applies a 3x3 rotation matrix and 3D translation to a PyVista mesh."""
    # Construct 4x4 affine transformation matrix
    t_mat = np.eye(4)
    t_mat[:3, :3] = rotation_matrix
    t_mat[:3, 3] = position

    return mesh.transform(t_mat, inplace=False)
