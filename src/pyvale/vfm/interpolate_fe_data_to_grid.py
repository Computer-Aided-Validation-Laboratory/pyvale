from __future__ import annotations

"""Interpolate FE centroid fields onto a regular grid for VFM preparation."""

import csv
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np
from scipy.interpolate import LinearNDInterpolator, NearestNDInterpolator
from scipy.spatial import Delaunay, cKDTree
from shapely import contains_xy
from shapely.affinity import scale as scale_geometry
from shapely.geometry import MultiPolygon, Polygon
from shapely.geometry.base import BaseGeometry
from shapely.ops import unary_union


@dataclass(slots=True, frozen=True)
class FePointCloudData:
    element_ids: np.ndarray
    time_values: np.ndarray
    coordinates_x: np.ndarray
    coordinates_y: np.ndarray
    component_names: tuple[str, ...]
    component_values: np.ndarray  # shape (point, component, timestep)


@dataclass(slots=True, frozen=True)
class FeInterpolatedGrid:
    x_grid: np.ndarray
    y_grid: np.ndarray
    strain: np.ndarray  # shape (timestep, component, y, x)
    specimen_mask: np.ndarray
    total_specimen_area: float | None
    metadata: dict[str, Any]


def load_fe_point_cloud_data(
    element_data_path: str | Path,
    *,
    component_columns: Sequence[str],
) -> FePointCloudData:
    """Load an ANSYS-style element-centroid table."""

    path = Path(element_data_path)
    with path.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError(f"FE element table '{path}' is empty.")

        required_columns = {"element_id", "time", "x", "y", *component_columns}
        missing_columns = sorted(required_columns.difference(reader.fieldnames))
        if missing_columns:
            raise ValueError(
                f"FE element table '{path}' is missing required columns: {missing_columns}."
            )

        rows = list(reader)

    if not rows:
        raise ValueError(f"FE element table '{path}' contains no data rows.")

    raw_element_ids = np.asarray([int(row["element_id"]) for row in rows], dtype=np.int64)
    raw_time_values = np.asarray([float(row["time"]) for row in rows], dtype=np.float64)
    raw_x = np.asarray([float(row["x"]) for row in rows], dtype=np.float64)
    raw_y = np.asarray([float(row["y"]) for row in rows], dtype=np.float64)
    raw_components = {
        column: np.asarray([float(row[column]) for row in rows], dtype=np.float64)
        for column in component_columns
    }

    element_ids = np.unique(raw_element_ids)
    time_values = np.unique(raw_time_values)
    point_count = element_ids.size
    timestep_count = time_values.size
    component_count = len(component_columns)

    element_index = {int(element_id): index for index, element_id in enumerate(element_ids)}
    time_index = {float(time_value): index for index, time_value in enumerate(time_values)}

    coordinates_x = np.full(point_count, np.nan, dtype=np.float64)
    coordinates_y = np.full(point_count, np.nan, dtype=np.float64)
    component_values = np.full((point_count, component_count, timestep_count), np.nan, dtype=np.float64)

    for row_index in range(raw_element_ids.size):
        point_index = element_index[int(raw_element_ids[row_index])]
        timestep_index = time_index[float(raw_time_values[row_index])]

        if np.isfinite(component_values[point_index, 0, timestep_index]):
            raise ValueError(
                "The FE element table contains duplicate (element_id, time) rows. "
                f"First duplicate at element_id={raw_element_ids[row_index]}, time={raw_time_values[row_index]:.9g}."
            )

        x_value = float(raw_x[row_index])
        y_value = float(raw_y[row_index])
        if np.isfinite(coordinates_x[point_index]):
            if not np.isclose(coordinates_x[point_index], x_value) or not np.isclose(
                coordinates_y[point_index], y_value
            ):
                raise ValueError(
                    "One element_id appears with different centroid coordinates across timesteps. "
                    f"element_id={raw_element_ids[row_index]}."
                )
        else:
            coordinates_x[point_index] = x_value
            coordinates_y[point_index] = y_value

        for component_index, component_name in enumerate(component_columns):
            component_values[point_index, component_index, timestep_index] = raw_components[component_name][row_index]

    if np.any(~np.isfinite(coordinates_x)) or np.any(~np.isfinite(coordinates_y)):
        raise ValueError(f"Some FE elements in '{path}' never received centroid coordinates.")
    if np.any(~np.isfinite(component_values)):
        raise ValueError(
            f"Some FE component values are missing in '{path}'. "
            "Expected a complete element-by-time table."
        )

    return FePointCloudData(
        element_ids=element_ids,
        time_values=time_values,
        coordinates_x=coordinates_x,
        coordinates_y=coordinates_y,
        component_names=tuple(component_columns),
        component_values=component_values,
    )


def interpolate_fe_data_to_grid(
    element_data_path: str | Path,
    *,
    component_columns: Sequence[str],
    mesh_path: str | Path | None = None,
    upsample_factor: float = 2.0,
    target_spacing: float | None = None,
) -> FeInterpolatedGrid:
    """Interpolate FE centroid data onto a regular physical grid."""

    point_cloud = load_fe_point_cloud_data(
        element_data_path,
        component_columns=component_columns,
    )

    points_xy = np.column_stack((point_cloud.coordinates_x, point_cloud.coordinates_y))
    geometry = build_surface_geometry_from_gmsh(mesh_path) if mesh_path is not None else None
    geometry_metadata: dict[str, Any] = {"mesh_path": str(mesh_path) if mesh_path is not None else None}
    if geometry is not None:
        geometry, geometry_metadata = _match_geometry_scale_to_points(geometry, points_xy, mesh_path=mesh_path)

    if points_xy.shape[0] == 1:
        strain = np.transpose(point_cloud.component_values, (2, 1, 0)).reshape(
            point_cloud.time_values.size,
            len(point_cloud.component_names),
            1,
            1,
        )
        return FeInterpolatedGrid(
            x_grid=points_xy[:, 0].reshape(1, 1).copy(),
            y_grid=points_xy[:, 1].reshape(1, 1).copy(),
            strain=strain,
            specimen_mask=np.ones((1, 1), dtype=bool),
            total_specimen_area=float(geometry.area) if geometry is not None else None,
            metadata={
                "element_data_path": str(element_data_path),
                "component_columns": list(component_columns),
                "raw_point_count": 1,
                "time_count": int(point_cloud.time_values.size),
                "grid_shape": [1, 1],
                "grid_spacing": None,
                "interpolation_method": "direct-single-point-copy",
                "nearest_fallback_point_count": 0,
                "warnings": [
                    "The FE export contains only one centroid, so no spatial interpolation was performed.",
                ],
                "geometry": geometry_metadata,
            },
        )

    grid_spacing = float(target_spacing) if target_spacing is not None else _estimate_grid_spacing(
        points_xy,
        upsample_factor=upsample_factor,
    )
    x_min, y_min, x_max, y_max = (
        geometry.bounds if geometry is not None else _point_cloud_bounds(points_xy)
    )
    x_axis = _build_regular_axis(x_min, x_max, grid_spacing)
    y_axis = _build_regular_axis(y_min, y_max, grid_spacing)
    x_grid, y_grid = np.meshgrid(x_axis, y_axis)

    specimen_mask = (
        _sample_geometry_mask(geometry, x_grid, y_grid)
        if geometry is not None
        else np.ones(x_grid.shape, dtype=bool)
    )

    flattened_fields = np.transpose(point_cloud.component_values, (0, 2, 1)).reshape(
        points_xy.shape[0],
        point_cloud.time_values.size * len(point_cloud.component_names),
    )

    triangulation = Delaunay(points_xy)
    linear_interpolator = LinearNDInterpolator(triangulation, flattened_fields, fill_value=np.nan)
    interpolated = np.asarray(linear_interpolator(x_grid, y_grid), dtype=np.float64)
    if interpolated.ndim == 2:
        interpolated = interpolated[:, :, None]

    inside_specimen_with_nan = specimen_mask & np.any(~np.isfinite(interpolated), axis=2)
    nearest_fallback_count = int(np.count_nonzero(inside_specimen_with_nan))
    if nearest_fallback_count > 0:
        nearest_interpolator = NearestNDInterpolator(points_xy, flattened_fields)
        fallback_values = np.asarray(
            nearest_interpolator(
                x_grid[inside_specimen_with_nan],
                y_grid[inside_specimen_with_nan],
            ),
            dtype=np.float64,
        )
        if fallback_values.ndim == 1:
            fallback_values = fallback_values[:, None]
        interpolated[inside_specimen_with_nan] = fallback_values

    interpolated = interpolated.reshape(
        x_grid.shape[0],
        x_grid.shape[1],
        point_cloud.time_values.size,
        len(point_cloud.component_names),
    )
    strain = np.transpose(interpolated, (2, 3, 0, 1))
    strain[:, :, ~specimen_mask] = np.nan

    x_output = np.asarray(x_grid, dtype=np.float64).copy()
    y_output = np.asarray(y_grid, dtype=np.float64).copy()
    x_output[~specimen_mask] = np.nan
    y_output[~specimen_mask] = np.nan

    warnings: list[str] = []
    if nearest_fallback_count > 0:
        warnings.append(
            "Nearest-neighbour fallback filled some regular-grid points that lay inside the specimen but outside "
            "the linear-interpolation support."
        )

    return FeInterpolatedGrid(
        x_grid=x_output,
        y_grid=y_output,
        strain=strain,
        specimen_mask=specimen_mask,
        total_specimen_area=float(geometry.area) if geometry is not None else None,
        metadata={
            "element_data_path": str(element_data_path),
            "component_columns": list(component_columns),
            "raw_point_count": int(points_xy.shape[0]),
            "time_count": int(point_cloud.time_values.size),
            "grid_shape": [int(x_grid.shape[0]), int(x_grid.shape[1])],
            "grid_spacing": float(grid_spacing),
            "interpolation_method": "linear-then-nearest-inside-specimen",
            "nearest_fallback_point_count": nearest_fallback_count,
            "warnings": warnings,
            "geometry": geometry_metadata,
        },
    )


def build_surface_geometry_from_gmsh(mesh_path: str | Path) -> BaseGeometry:
    """Build the unioned 2D specimen geometry from a Gmsh mesh file."""

    path = Path(mesh_path)
    lines = path.read_text(encoding="utf-8").splitlines()
    if "$MeshFormat" not in lines:
        raise ValueError(f"Mesh file '{path}' does not look like a Gmsh ASCII mesh.")

    mesh_format_index = lines.index("$MeshFormat")
    version = lines[mesh_format_index + 1].split()[0]
    major_version = int(version.split(".")[0])
    if major_version == 2:
        polygons = _parse_gmsh_v22_surface_polygons(lines)
    elif major_version == 4:
        polygons = _parse_gmsh_v41_surface_polygons(lines)
    else:
        raise ValueError(f"Unsupported Gmsh major version '{major_version}' in '{path}'.")

    if not polygons:
        raise ValueError(f"No 2D surface elements were found in mesh '{path}'.")

    geometry = unary_union(polygons).buffer(0.0)
    if geometry.is_empty:
        raise ValueError(f"The unioned specimen geometry from '{path}' is empty.")
    return geometry


def _parse_gmsh_v22_surface_polygons(lines: list[str]) -> list[Polygon]:
    nodes_start = lines.index("$Nodes")
    node_count = int(lines[nodes_start + 1].strip())
    nodes: dict[int, tuple[float, float]] = {}
    for line in lines[nodes_start + 2 : nodes_start + 2 + node_count]:
        parts = line.split()
        node_id = int(parts[0])
        nodes[node_id] = (float(parts[1]), float(parts[2]))

    elements_start = lines.index("$Elements")
    element_count = int(lines[elements_start + 1].strip())
    polygons: list[Polygon] = []
    for line in lines[elements_start + 2 : elements_start + 2 + element_count]:
        parts = line.split()
        element_type = int(parts[1])
        if element_type not in _SUPPORTED_SURFACE_ELEMENT_TYPES:
            continue
        tag_count = int(parts[2])
        node_ids = [int(node_id) for node_id in parts[3 + tag_count :]]
        polygon = _polygon_from_surface_element(nodes, element_type, node_ids)
        if polygon is not None:
            polygons.append(polygon)
    return polygons


def _parse_gmsh_v41_surface_polygons(lines: list[str]) -> list[Polygon]:
    nodes_start = lines.index("$Nodes")
    nodes_header = [int(value) for value in lines[nodes_start + 1].split()]
    node_block_count = nodes_header[0]
    line_index = nodes_start + 2
    nodes: dict[int, tuple[float, float]] = {}
    for _ in range(node_block_count):
        _, _, _, node_count_in_block = map(int, lines[line_index].split())
        line_index += 1
        node_ids = [int(lines[line_index + offset].strip()) for offset in range(node_count_in_block)]
        line_index += node_count_in_block
        for node_id in node_ids:
            x_coord, y_coord, _ = map(float, lines[line_index].split())
            nodes[node_id] = (x_coord, y_coord)
            line_index += 1

    elements_start = lines.index("$Elements")
    element_header = [int(value) for value in lines[elements_start + 1].split()]
    element_block_count = element_header[0]
    line_index = elements_start + 2
    polygons: list[Polygon] = []
    for _ in range(element_block_count):
        entity_dim, _, element_type, element_count_in_block = map(int, lines[line_index].split())
        line_index += 1
        for _ in range(element_count_in_block):
            parts = lines[line_index].split()
            line_index += 1
            if entity_dim != 2 or element_type not in _SUPPORTED_SURFACE_ELEMENT_TYPES:
                continue
            node_ids = [int(node_id) for node_id in parts[1:]]
            polygon = _polygon_from_surface_element(nodes, element_type, node_ids)
            if polygon is not None:
                polygons.append(polygon)
    return polygons


_SUPPORTED_SURFACE_ELEMENT_TYPES = {2, 3, 9, 10, 16}


def _polygon_from_surface_element(
    nodes: dict[int, tuple[float, float]],
    element_type: int,
    node_ids: Sequence[int],
) -> Polygon | None:
    boundary_node_ids = _surface_boundary_node_ids(element_type, node_ids)
    coordinates = [nodes[node_id] for node_id in boundary_node_ids]
    polygon = Polygon(coordinates)
    if polygon.is_empty or polygon.area <= 0.0:
        polygon = polygon.buffer(0.0)
    if polygon.is_empty or polygon.area <= 0.0:
        return None
    return polygon


def _surface_boundary_node_ids(element_type: int, node_ids: Sequence[int]) -> list[int]:
    if element_type == 2:
        return [int(node_id) for node_id in node_ids[:3]]
    if element_type == 3:
        return [int(node_id) for node_id in node_ids[:4]]
    if element_type == 9:
        order = (0, 3, 1, 4, 2, 5)
        return [int(node_ids[index]) for index in order]
    if element_type in {10, 16}:
        order = (0, 4, 1, 5, 2, 6, 3, 7)
        return [int(node_ids[index]) for index in order]
    raise ValueError(f"Unsupported surface element type '{element_type}'.")


def _estimate_grid_spacing(points_xy: np.ndarray, *, upsample_factor: float) -> float:
    if upsample_factor <= 0.0:
        raise ValueError(f"upsample_factor must be positive, got {upsample_factor}.")

    tree = cKDTree(points_xy)
    distances, _ = tree.query(points_xy, k=2)
    nearest_neighbour_distances = distances[:, 1]
    positive_distances = nearest_neighbour_distances[nearest_neighbour_distances > 0.0]
    if positive_distances.size == 0:
        raise ValueError("Could not estimate a representative FE point spacing from duplicate coordinates.")
    return float(np.median(positive_distances) / upsample_factor)


def _build_regular_axis(min_value: float, max_value: float, spacing: float) -> np.ndarray:
    if spacing <= 0.0:
        raise ValueError(f"Regular-grid spacing must be positive, got {spacing}.")

    span = float(max_value - min_value)
    if span <= 0.0:
        return np.asarray([float(min_value)], dtype=np.float64)

    interval_count = max(1, int(math.ceil(span / spacing)))
    return np.linspace(min_value, max_value, interval_count + 1, dtype=np.float64)


def _point_cloud_bounds(points_xy: np.ndarray) -> tuple[float, float, float, float]:
    x_min = float(np.min(points_xy[:, 0]))
    y_min = float(np.min(points_xy[:, 1]))
    x_max = float(np.max(points_xy[:, 0]))
    y_max = float(np.max(points_xy[:, 1]))
    return x_min, y_min, x_max, y_max


def _sample_geometry_mask(
    geometry: BaseGeometry,
    x_grid: np.ndarray,
    y_grid: np.ndarray,
) -> np.ndarray:
    bounds = geometry.bounds
    tolerance = max(
        1.0e-12,
        1.0e-9 * max(bounds[2] - bounds[0], bounds[3] - bounds[1], 1.0),
    )
    return np.asarray(contains_xy(geometry.buffer(tolerance), x_grid, y_grid), dtype=bool)


def _match_geometry_scale_to_points(
    geometry: BaseGeometry,
    points_xy: np.ndarray,
    *,
    mesh_path: str | Path | None,
) -> tuple[BaseGeometry, dict[str, Any]]:
    candidate_scales = (1.0e-6, 1.0e-3, 1.0e-2, 1.0e-1, 1.0, 10.0, 100.0, 1000.0, 1.0e6)
    point_span = max(
        float(np.ptp(points_xy[:, 0])),
        float(np.ptp(points_xy[:, 1])),
        1.0,
    )
    tolerance = max(1.0e-12, 1.0e-6 * point_span)

    def _count_points_inside(candidate: BaseGeometry) -> int:
        return int(np.count_nonzero(contains_xy(candidate.buffer(tolerance), points_xy[:, 0], points_xy[:, 1])))

    base_count = _count_points_inside(geometry)
    best_scale = 1.0
    best_geometry = geometry
    best_count = base_count

    for scale_factor in candidate_scales:
        candidate = scale_geometry(geometry, xfact=scale_factor, yfact=scale_factor, origin=(0.0, 0.0))
        count = _count_points_inside(candidate)
        if count > best_count:
            best_scale = float(scale_factor)
            best_geometry = candidate
            best_count = count

    applied_scale = 1.0
    resolved_geometry = geometry
    warnings: list[str] = []
    if best_scale != 1.0 and best_count >= max(int(0.9 * points_xy.shape[0]), base_count + 1):
        resolved_geometry = best_geometry
        applied_scale = best_scale
        warnings.append(
            "Applied a uniform scale factor to the Gmsh geometry so it matches the FE centroid coordinate units."
        )

    return resolved_geometry, {
        "mesh_path": str(mesh_path) if mesh_path is not None else None,
        "applied_scale_factor": float(applied_scale),
        "points_inside_unscaled_geometry": base_count,
        "points_inside_scaled_geometry": best_count,
        "warnings": warnings,
    }
