from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import numpy.typing as npt
from shapely.geometry import GeometryCollection, MultiPolygon, Polygon
from shapely.geometry.base import BaseGeometry

from pyvale.vfm.experimentdata import SpecimenGeometry
from pyvale.vfm.spatialparamslicewise import (
    SliceAxis,
    SliceConfig,
    SlicePartition,
    _associate_points_to_slices,
    _build_slice_band,
    _resolve_slice_boundaries,
    build_slice_partition,
)
from pyvale.vfm.vfmregionofinterest import build_roi_geometry
from pyvale.vfm.vfmesh import _generate_data_mesh_nodal_coord


_GEOMETRY_TOLERANCE = 1.0e-9


@dataclass(slots=True, frozen=True)
class SliceAreaPartition:
    """Slice partition backed by precomputed DIC support-cell overlap areas.

    `areas` and `widths` are the discrete operator values implied by the
    support-cell overlaps. `geometric_areas` and `geometric_widths` are the
    exact ROI/slice polygon values used for diagnostics.
    """

    axis: SliceAxis
    boundaries: npt.NDArray[np.float64]
    centres: npt.NDArray[np.float64]
    spans: npt.NDArray[np.float64]
    widths: npt.NDArray[np.float64]
    areas: npt.NDArray[np.float64]
    geometric_widths: npt.NDArray[np.float64]
    geometric_areas: npt.NDArray[np.float64]
    coverage_fractions: npt.NDArray[np.float64]
    point_counts: npt.NDArray[np.int64]
    slice_id_map: npt.NDArray[np.int32]
    valid_point_mask: npt.NDArray[np.bool_]
    slice_point_indices: tuple[npt.NDArray[np.int64], ...]
    slice_force_point_indices: tuple[npt.NDArray[np.int64], ...]
    slice_force_point_areas: tuple[npt.NDArray[np.float64], ...]
    slice_force_point_weights: tuple[npt.NDArray[np.float64], ...]
    support_node_x: npt.NDArray[np.float64]
    support_node_y: npt.NDArray[np.float64]
    slice_geometries: tuple[BaseGeometry, ...]
    num_slices: int

    def get_slice_mask(self, slice_index: int) -> npt.NDArray[np.bool_]:
        return self.slice_id_map == slice_index

    def get_slice_active_cell_mask(self, slice_index: int) -> npt.NDArray[np.bool_]:
        if slice_index < 0 or slice_index >= self.num_slices:
            raise IndexError(f"Slice index {slice_index} is out of range.")

        active_mask = np.zeros(self.slice_id_map.shape, dtype=bool)
        active_indices = self.slice_force_point_indices[slice_index]
        if active_indices.size > 0:
            active_mask.ravel()[active_indices] = True
        return active_mask


@dataclass(slots=True, frozen=True)
class SliceAreaComparison:
    """Compare slice areas from different geometric/integration viewpoints."""

    slice_partition: SlicePartition
    area_partition: SliceAreaPartition
    point_area_sums: npt.NDArray[np.float64]
    line_integral_areas: npt.NDArray[np.float64] = field(init=False)
    polygon_areas: npt.NDArray[np.float64] = field(init=False)
    overlap_areas: npt.NDArray[np.float64] = field(init=False)
    spans: npt.NDArray[np.float64] = field(init=False)
    line_integral_widths: npt.NDArray[np.float64] = field(init=False)
    polygon_widths: npt.NDArray[np.float64] = field(init=False)
    overlap_widths: npt.NDArray[np.float64] = field(init=False)
    point_area_widths: npt.NDArray[np.float64] = field(init=False)
    line_minus_polygon_area: npt.NDArray[np.float64] = field(init=False)
    overlap_minus_polygon_area: npt.NDArray[np.float64] = field(init=False)
    point_minus_polygon_area: npt.NDArray[np.float64] = field(init=False)
    line_vs_polygon_relative_error: npt.NDArray[np.float64] = field(init=False)
    overlap_vs_polygon_relative_error: npt.NDArray[np.float64] = field(init=False)
    point_vs_polygon_relative_error: npt.NDArray[np.float64] = field(init=False)

    def __post_init__(self) -> None:
        spans = self.slice_partition.spans
        line_integral_areas = self.slice_partition.areas
        polygon_areas = self.area_partition.geometric_areas
        overlap_areas = self.area_partition.areas
        line_integral_widths = self.slice_partition.widths
        polygon_widths = self.area_partition.geometric_widths
        overlap_widths = self.area_partition.widths
        point_area_widths = np.divide(
            self.point_area_sums,
            spans,
            out=np.zeros_like(self.point_area_sums),
            where=spans > 0.0,
        )

        object.__setattr__(self, "line_integral_areas", line_integral_areas)
        object.__setattr__(self, "polygon_areas", polygon_areas)
        object.__setattr__(self, "overlap_areas", overlap_areas)
        object.__setattr__(self, "spans", spans)
        object.__setattr__(self, "line_integral_widths", line_integral_widths)
        object.__setattr__(self, "polygon_widths", polygon_widths)
        object.__setattr__(self, "overlap_widths", overlap_widths)
        object.__setattr__(self, "point_area_widths", point_area_widths)
        object.__setattr__(self, "line_minus_polygon_area", line_integral_areas - polygon_areas)
        object.__setattr__(self, "overlap_minus_polygon_area", overlap_areas - polygon_areas)
        object.__setattr__(self, "point_minus_polygon_area", self.point_area_sums - polygon_areas)
        object.__setattr__(self, "line_vs_polygon_relative_error", _safe_relative_error(line_integral_areas, polygon_areas))
        object.__setattr__(self, "overlap_vs_polygon_relative_error", _safe_relative_error(overlap_areas, polygon_areas))
        object.__setattr__(self, "point_vs_polygon_relative_error", _safe_relative_error(self.point_area_sums, polygon_areas))


def build_slice_area_partition(
    specimen_geometry: SpecimenGeometry,
    slice_config: SliceConfig | None = None,
    plot_diagnostic: bool = False,
    diagnostic_slice_index: int | None = None,
) -> SliceAreaPartition:
    """Build an area-based slice partition from DIC support-cell overlaps."""

    if slice_config is None:
        raise ValueError("slice_config must be provided explicitly for area-based slice-wise partitioning.")
    config = slice_config

    valid_point_mask = (
        specimen_geometry.specimen_mask
        & np.isfinite(specimen_geometry.x)
        & np.isfinite(specimen_geometry.y)
        & np.isfinite(specimen_geometry.pixel_area)
    )
    if not np.any(valid_point_mask):
        raise ValueError("No valid specimen points were available to build the area-based slice partition.")

    # Determine slice boundaries along the chosen axis
    coordinate_along = specimen_geometry.x if config.axis == "x" else specimen_geometry.y
    valid_coordinates_along = coordinate_along[valid_point_mask]
    boundaries = _resolve_slice_boundaries(
        valid_coordinates_along,
        config.boundaries,
        config.num_slices,
    )

    # Compute slice spans and centres along chosen axis
    spans = np.diff(boundaries)
    centres = 0.5 * (boundaries[:-1] + boundaries[1:])
    
    # Build ROI geometry from definition
    roi_geometry = build_roi_geometry(specimen_geometry.region_of_interest.roi_definition)

    # Build slice geometries (polygon intersection of ROI and slices) and compute geometric areas
    slice_geometries, geometric_areas = _build_slice_geometries(
        roi_geometry=roi_geometry,
        axis=config.axis,
        boundaries=boundaries,
    )

    # Compute average geometric widths for each slice (geometric area / span)
    geometric_widths = np.divide(
        geometric_areas,
        spans,
        out=np.zeros_like(geometric_areas),
        where=spans > 0.0,
    )

    # The current metric treats the native DIC support cells as the source of
    # truth for the stress field and assumes the DIC data extends to the ROI boundary. 
    # 
    # A future option could instead derive the supportregion directly from the ROI 
    # geometry and then map/interpolate data onto it. This could improve accuracy 
    # for cases in which DIC data is sparse or doesn't fully cover the ROI, 
    # but would require interpolation and weighting of stress field.
    
    # Assemble data mesh nodal coordinates for the DIC support cells
    support_node_x, support_node_y = _build_support_node_grids(specimen_geometry)

    # Compute the overlap areas, weights, and indices of DIC support cells (DIC datapoint elements) for each slice
    (
        slice_force_point_indices,
        slice_force_point_areas,
        slice_force_point_weights,
        discrete_areas,
    ) = _build_slice_support_overlap_operator(
        specimen_geometry=specimen_geometry,
        axis=config.axis,
        boundaries=boundaries,
        spans=spans,
        valid_point_mask=valid_point_mask,
        support_node_x=support_node_x,
        support_node_y=support_node_y,
        slice_geometries=slice_geometries,
        geometric_areas=geometric_areas,
    )

    # Compute average discrete widths for each slice (discrete area / span)
    widths = np.divide(
        discrete_areas,
        spans,
        out=np.zeros_like(discrete_areas),
        where=spans > 0.0,
    )

    # Compute coverage fractions for each slice (discrete area / geometric area)
    coverage_fractions = np.divide(
        discrete_areas,
        geometric_areas,
        out=np.ones_like(discrete_areas),
        where=geometric_areas > _GEOMETRY_TOLERANCE,
    )

    # Associate valid DIC support cells (DIC datapoint elements) to slices and count them
    slice_id_map, slice_point_indices, point_counts = _associate_points_to_slices(
        specimen_geometry,
        axis=config.axis,
        boundaries=boundaries,
        valid_point_mask=valid_point_mask,
    )

    partition = SliceAreaPartition(
        axis=config.axis,
        boundaries=boundaries,
        centres=centres,
        spans=spans,
        widths=widths,
        areas=discrete_areas,
        geometric_widths=geometric_widths,
        geometric_areas=geometric_areas,
        coverage_fractions=coverage_fractions,
        point_counts=point_counts,
        slice_id_map=slice_id_map,
        valid_point_mask=valid_point_mask,
        slice_point_indices=slice_point_indices,
        slice_force_point_indices=slice_force_point_indices,
        slice_force_point_areas=slice_force_point_areas,
        slice_force_point_weights=slice_force_point_weights,
        support_node_x=support_node_x,
        support_node_y=support_node_y,
        slice_geometries=slice_geometries,
        num_slices=int(spans.size),
    )

    if plot_diagnostic:
        plot_slice_area_partition_diagnostic(
            specimen_geometry,
            partition,
            slice_index=diagnostic_slice_index,
        )

    return partition


def plot_slice_area_partition_diagnostic(
    specimen_geometry: SpecimenGeometry,
    slice_partition: SliceAreaPartition,
    slice_index: int | None = None,
) -> None:
    """Plot the ROI, slices, and DIC support cells used by the area operator."""

    import matplotlib.pyplot as plt
    from matplotlib.collections import PolyCollection

    fig, ax = plt.subplots()

    valid_rows, valid_cols = np.nonzero(slice_partition.valid_point_mask)
    active_mask = (
        slice_partition.get_slice_active_cell_mask(slice_index)
        if slice_index is not None
        else np.zeros(slice_partition.valid_point_mask.shape, dtype=bool)
    )

    inactive_polygons: list[npt.NDArray[np.float64]] = []
    active_polygons: list[npt.NDArray[np.float64]] = []
    for row_index, col_index in zip(valid_rows, valid_cols, strict=True):
        vertices = _build_support_cell_vertices(
            slice_partition.support_node_x,
            slice_partition.support_node_y,
            int(row_index),
            int(col_index),
        )
        if slice_index is not None and active_mask[row_index, col_index]:
            active_polygons.append(vertices)
        else:
            inactive_polygons.append(vertices)

    if inactive_polygons:
        ax.add_collection(
            PolyCollection(
                inactive_polygons,
                facecolors="none",
                edgecolors="#b0b7bf",
                linewidths=0.45,
            )
        )
    if active_polygons:
        ax.add_collection(
            PolyCollection(
                active_polygons,
                facecolors="#4C78A8",
                edgecolors="#1f4e79",
                linewidths=0.7,
                alpha=0.32,
            )
        )

    roi_geometry = build_roi_geometry(specimen_geometry.region_of_interest.roi_definition)
    _plot_geometry_boundaries(ax, roi_geometry, color="black", linewidth=1.6)

    colours = ("#F58518", "#54A24B", "#E45756", "#72B7B2")
    for current_slice_index, geometry in enumerate(slice_partition.slice_geometries):
        colour = colours[current_slice_index % len(colours)]
        linewidth = 2.0 if current_slice_index == slice_index else 1.0
        linestyle = "-" if current_slice_index == slice_index else "--"
        _plot_geometry_boundaries(ax, geometry, color=colour, linewidth=linewidth, linestyle=linestyle)

    ax.scatter(
        specimen_geometry.x[slice_partition.valid_point_mask],
        specimen_geometry.y[slice_partition.valid_point_mask],
        s=5,
        c="black",
        alpha=0.35,
        linewidths=0.0,
        zorder=4,
    )

    if slice_index is None:
        ax.set_title("Area-based slice partition diagnostic")
    else:
        coverage = float(slice_partition.coverage_fractions[slice_index])
        ax.set_title(
            "Area-based slice partition diagnostic "
            f"(slice {slice_index}, coverage={coverage:.3f})"
        )

    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    fig.tight_layout()


def compare_slice_areas(
    specimen_geometry: SpecimenGeometry,
    slice_config: SliceConfig,
) -> SliceAreaComparison:
    """Compare line-based, polygon, overlap-area, and point-area slice areas."""

    line_partition = build_slice_partition(
        specimen_geometry,
        slice_config=slice_config,
        plot_diagnostic=False,
    )
    area_partition = build_slice_area_partition(
        specimen_geometry,
        slice_config=slice_config,
        plot_diagnostic=False,
    )
    point_area_sums = _compute_point_area_sums(
        specimen_geometry=specimen_geometry,
        slice_partition=line_partition,
    )
    return SliceAreaComparison(
        slice_partition=line_partition,
        area_partition=area_partition,
        point_area_sums=point_area_sums,
    )


def summarise_slice_area_comparison(comparison: SliceAreaComparison) -> str:
    """Return a compact text table for terminal inspection."""

    header = (
        "slice "
        "line_area "
        "polygon_area "
        "overlap_area "
        "point_area "
        "line_rel_err "
        "overlap_rel_err "
        "point_rel_err"
    )
    lines = [header]
    for slice_index in range(comparison.slice_partition.num_slices):
        lines.append(
            (
                f"{slice_index:>5d} "
                f"{comparison.line_integral_areas[slice_index]:>10.6f} "
                f"{comparison.polygon_areas[slice_index]:>12.6f} "
                f"{comparison.overlap_areas[slice_index]:>12.6f} "
                f"{comparison.point_area_sums[slice_index]:>10.6f} "
                f"{comparison.line_vs_polygon_relative_error[slice_index]:>12.6f} "
                f"{comparison.overlap_vs_polygon_relative_error[slice_index]:>15.6f} "
                f"{comparison.point_vs_polygon_relative_error[slice_index]:>13.6f}"
            )
        )

    max_line_error = float(np.max(np.abs(comparison.line_vs_polygon_relative_error)))
    max_overlap_error = float(np.max(np.abs(comparison.overlap_vs_polygon_relative_error)))
    max_point_error = float(np.max(np.abs(comparison.point_vs_polygon_relative_error)))
    lines.append("")
    lines.append(f"max |line rel err|    = {max_line_error:.6f}")
    lines.append(f"max |overlap rel err| = {max_overlap_error:.6f}")
    lines.append(f"max |point rel err|   = {max_point_error:.6f}")
    return "\n".join(lines)


def _build_slice_geometries(
    *,
    roi_geometry: BaseGeometry,
    axis: SliceAxis,
    boundaries: npt.NDArray[np.float64],
) -> tuple[tuple[BaseGeometry, ...], npt.NDArray[np.float64]]:
    bounds = roi_geometry.bounds
    if axis == "y":
        cross_min = float(bounds[0])
        cross_max = float(bounds[2])
    else:
        cross_min = float(bounds[1])
        cross_max = float(bounds[3])
    cross_pad = max(0.05 * (cross_max - cross_min), 1.0)

    slice_geometries: list[BaseGeometry] = []
    geometric_areas = np.zeros(boundaries.size - 1, dtype=np.float64)
    for slice_index in range(boundaries.size - 1):
        slice_band = _build_slice_band(
            axis=axis,
            along_min=float(boundaries[slice_index]),
            along_max=float(boundaries[slice_index + 1]),
            cross_min=cross_min,
            cross_max=cross_max,
            cross_pad=cross_pad,
        )
        geometry = roi_geometry.intersection(slice_band)
        slice_geometries.append(geometry)
        geometric_areas[slice_index] = float(geometry.area)

    return tuple(slice_geometries), geometric_areas


def _build_support_node_grids(
    specimen_geometry: SpecimenGeometry,
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    data_mesh = _generate_data_mesh_nodal_coord(
        specimen_geometry.x,
        specimen_geometry.y,
    )
    return data_mesh.nodal_coord_x, data_mesh.nodal_coord_y


def _build_slice_support_overlap_operator(
    *,
    specimen_geometry: SpecimenGeometry,
    axis: SliceAxis,
    boundaries: npt.NDArray[np.float64],
    spans: npt.NDArray[np.float64],
    valid_point_mask: npt.NDArray[np.bool_],
    support_node_x: npt.NDArray[np.float64],
    support_node_y: npt.NDArray[np.float64],
    slice_geometries: tuple[BaseGeometry, ...],
    geometric_areas: npt.NDArray[np.float64],
) -> tuple[
    tuple[npt.NDArray[np.int64], ...],
    tuple[npt.NDArray[np.float64], ...],
    tuple[npt.NDArray[np.float64], ...],
    npt.NDArray[np.float64],
]:
    num_slices = spans.size
    slice_index_lists: list[list[int]] = [[] for _ in range(num_slices)]
    slice_area_lists: list[list[float]] = [[] for _ in range(num_slices)]
    slice_weight_lists: list[list[float]] = [[] for _ in range(num_slices)]
    discrete_areas = np.zeros(num_slices, dtype=np.float64)

    valid_rows, valid_cols = np.nonzero(valid_point_mask)
    # Loop through all valid DIC support cells (DIC data point elements)
    for row_index, col_index in zip(valid_rows, valid_cols, strict=True):
        row_index = int(row_index)
        col_index = int(col_index)
        
        # Build polygon for current support cell and check validity/area
        cell_vertices = _build_support_cell_vertices(
            support_node_x,
            support_node_y,
            row_index,
            col_index,
        )
        cell_polygon = Polygon(cell_vertices)
        if not cell_polygon.is_valid:
            cell_polygon = cell_polygon.buffer(0.0)
        if cell_polygon.is_empty or cell_polygon.area <= _GEOMETRY_TOLERANCE:
            continue

        # Determine the slices that the current support cell overlaps based on its min/max coordinates along the slicing axis
        along_coordinates = cell_vertices[:, 0] if axis == "x" else cell_vertices[:, 1]
        along_min = float(np.min(along_coordinates))
        along_max = float(np.max(along_coordinates))

        slice_start = int(np.searchsorted(boundaries, along_min, side="right") - 1)
        slice_end = int(np.searchsorted(boundaries, along_max, side="left") - 1)
        if slice_end < 0 or slice_start > num_slices - 1:
            continue

        slice_start = max(0, slice_start)
        slice_end = min(num_slices - 1, slice_end)
        
        # Compute the flat index of the current support cell in the original 2D grid
        flat_index = int(np.ravel_multi_index((row_index, col_index), specimen_geometry.x.shape))

        # Loop through the overlapping slices and compute the overlap area, weight, and store indices
        for slice_index in range(slice_start, slice_end + 1):
            if _interval_overlap(
                along_min,
                along_max,
                float(boundaries[slice_index]),
                float(boundaries[slice_index + 1]),
            ) <= _GEOMETRY_TOLERANCE:
                continue
            
            # Compute the overlap area between the support cell polygon and the slice geometry
            overlap_area = float(cell_polygon.intersection(slice_geometries[slice_index]).area)
            if overlap_area <= _GEOMETRY_TOLERANCE:
                continue

            slice_index_lists[slice_index].append(flat_index)
            slice_area_lists[slice_index].append(overlap_area)
            slice_weight_lists[slice_index].append(overlap_area / float(spans[slice_index]))
            discrete_areas[slice_index] += overlap_area

    for slice_index, geometric_area in enumerate(geometric_areas):
        if geometric_area <= _GEOMETRY_TOLERANCE:
            continue
        if len(slice_index_lists[slice_index]) == 0:
            raise ValueError(
                "A slice region has non-zero geometric area but no valid DIC support cells overlap it. "
                "Check the ROI/data alignment or refine the slice definition."
            )

    return (
        tuple(np.asarray(indices, dtype=np.int64) for indices in slice_index_lists),
        tuple(np.asarray(areas, dtype=np.float64) for areas in slice_area_lists),
        tuple(np.asarray(weights, dtype=np.float64) for weights in slice_weight_lists),
        discrete_areas,
    )


def _build_support_cell_vertices(
    support_node_x: npt.NDArray[np.float64],
    support_node_y: npt.NDArray[np.float64],
    row_index: int,
    col_index: int,
) -> npt.NDArray[np.float64]:
    return np.asarray(
        (
            (support_node_x[row_index, col_index], support_node_y[row_index, col_index]),
            (support_node_x[row_index, col_index + 1], support_node_y[row_index, col_index + 1]),
            (support_node_x[row_index + 1, col_index + 1], support_node_y[row_index + 1, col_index + 1]),
            (support_node_x[row_index + 1, col_index], support_node_y[row_index + 1, col_index]),
        ),
        dtype=np.float64,
    )


def _plot_geometry_boundaries(
    ax,
    geometry: BaseGeometry,
    *,
    color: str,
    linewidth: float,
    linestyle: str = "-",
) -> None:
    if geometry.is_empty:
        return

    if isinstance(geometry, Polygon):
        exterior = np.asarray(geometry.exterior.coords, dtype=np.float64)
        ax.plot(exterior[:, 0], exterior[:, 1], color=color, linewidth=linewidth, linestyle=linestyle)
        for interior in geometry.interiors:
            ring = np.asarray(interior.coords, dtype=np.float64)
            ax.plot(ring[:, 0], ring[:, 1], color=color, linewidth=linewidth, linestyle=linestyle)
        return

    if isinstance(geometry, MultiPolygon):
        for polygon in geometry.geoms:
            _plot_geometry_boundaries(
                ax,
                polygon,
                color=color,
                linewidth=linewidth,
                linestyle=linestyle,
            )
        return

    if isinstance(geometry, GeometryCollection):
        for item in geometry.geoms:
            _plot_geometry_boundaries(
                ax,
                item,
                color=color,
                linewidth=linewidth,
                linestyle=linestyle,
            )


def _compute_point_area_sums(
    *,
    specimen_geometry: SpecimenGeometry,
    slice_partition: SlicePartition,
) -> npt.NDArray[np.float64]:
    flat_pixel_area = specimen_geometry.pixel_area.ravel()
    point_area_sums = np.zeros(slice_partition.num_slices, dtype=np.float64)
    for slice_index, flat_indices in enumerate(slice_partition.slice_point_indices):
        if flat_indices.size == 0:
            continue
        point_area_sums[slice_index] = float(np.sum(flat_pixel_area[flat_indices]))
    return point_area_sums


def _interval_overlap(
    interval_min: float,
    interval_max: float,
    target_min: float,
    target_max: float,
) -> float:
    return max(0.0, min(interval_max, target_max) - max(interval_min, target_min))


def _safe_relative_error(
    values: npt.NDArray[np.float64],
    reference: npt.NDArray[np.float64],
) -> npt.NDArray[np.float64]:
    scale = np.where(np.abs(reference) > 0.0, reference, 1.0)
    return (values - reference) / scale
