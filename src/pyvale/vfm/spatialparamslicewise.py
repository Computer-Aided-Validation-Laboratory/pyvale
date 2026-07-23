from __future__ import annotations

from copy import copy
from dataclasses import dataclass
from typing import Literal

import numpy as np
import numpy.typing as npt
from shapely.geometry import GeometryCollection, LineString, MultiLineString, box
from shapely.geometry.base import BaseGeometry

from pyvale.vfm.constparam import ConstitutiveParameter
from pyvale.vfm.dof import DegreeOfFreedom
from pyvale.vfm.experimentdata import SpecimenGeometry
from pyvale.vfm.spatialparam import ISpatialParameterisation
from pyvale.vfm.vfmregionofinterest import build_roi_geometry


SliceAxis = Literal["x", "y"]
_GEOMETRY_TOLERANCE = 1.0e-9


@dataclass(slots=True, frozen=True)
class SliceConfig:
    """Configuration for slice-wise parameterisation.

    Axis must be provided explicitly. Provide either `num_slices` or
    `boundaries`. If both are provided, `boundaries` define the slices and
    `num_slices` is only used as a consistency check.
    """

    axis: SliceAxis
    num_slices: int | None = None
    boundaries: npt.NDArray[np.float64] | None = None

    def __post_init__(self) -> None:
        if self.axis not in ("x", "y"):
            raise ValueError(f"Unsupported slice axis '{self.axis}'. Expected 'x' or 'y'.")

        has_num_slices = self.num_slices is not None
        has_boundaries = self.boundaries is not None
        if not has_num_slices and not has_boundaries:
            raise ValueError("Provide either num_slices or boundaries when configuring slice-wise parameters.")
        if has_num_slices and self.num_slices is not None and self.num_slices < 1:
            raise ValueError("num_slices must be at least 1.")
        if has_num_slices and has_boundaries and self.boundaries is not None and self.num_slices is not None:
            if self.num_slices != len(self.boundaries) - 1:
                raise ValueError("num_slices must match len(boundaries) - 1 when both are provided.")


@dataclass(slots=True, frozen=True)
class SliceCrossSection:
    """Precomputed line-integral data for one cross-section within a slice."""

    position: float
    width_weight: float
    line_length: float
    segment_bounds: tuple[tuple[float, float], ...]
    point_coordinates: npt.NDArray[np.float64]
    point_cell_bounds: npt.NDArray[np.float64]
    point_indices: npt.NDArray[np.int64]
    point_area_integral_weights: npt.NDArray[np.float64]
    point_segment_ids: npt.NDArray[np.int32]


@dataclass(slots=True, frozen=True)
class SlicePartition:
    """Partition of the specimen into slice regions.

    The slice geometry is defined from the ROI as a geometric object, then
    discretised into a weighted set of cross-sectional line integrals. This
    means the reconstructed slice force is formed from the weighted average of
    cross-sectional line integrals rather than by directly summing all stress
    points inside the slice area.
    """

    axis: SliceAxis
    boundaries: npt.NDArray[np.float64]
    centres: npt.NDArray[np.float64]
    spans: npt.NDArray[np.float64]
    widths: npt.NDArray[np.float64]
    areas: npt.NDArray[np.float64]
    cross_centres: npt.NDArray[np.float64]
    cross_bounds: npt.NDArray[np.float64]
    point_counts: npt.NDArray[np.int64]
    slice_id_map: npt.NDArray[np.int32]
    valid_point_mask: npt.NDArray[np.bool_]
    slice_point_indices: tuple[npt.NDArray[np.int64], ...]
    cross_sections: tuple[tuple[SliceCrossSection, ...], ...]

    @property
    def num_slices(self) -> int:
        return int(self.spans.size)

    def get_slice_mask(self, slice_index: int) -> npt.NDArray[np.bool_]:
        return self.slice_id_map == slice_index


def build_slice_partition(
    specimen_geometry: SpecimenGeometry,
    slice_config: SliceConfig | None = None,
    plot_diagnostic: bool = False,
) -> SlicePartition:
    """Build the slice geometry and DIC point associations.

    The slice boundaries are resolved along the chosen axis, the ROI is
    converted to a geometric object, and each slice is represented by a
    weighted collection of cross-sectional line integrals through the ROI.
    """

    if slice_config is None:
        raise ValueError("slice_config must be provided explicitly for slice-wise parameterisation.")
    config = slice_config

    #TODO: x,y,pixel area should always be finite so may be redunant
    valid_point_mask = (
        specimen_geometry.specimen_mask
        & np.isfinite(specimen_geometry.x)
        & np.isfinite(specimen_geometry.y)
        & np.isfinite(specimen_geometry.pixel_area)
    )
    if not np.any(valid_point_mask):
        raise ValueError("No valid specimen points were available to build the slice partition.")

    # Resolve coordinates along slicing axis
    coordinate_along = specimen_geometry.x if config.axis == "x" else specimen_geometry.y
    valid_coordinates_along = coordinate_along[valid_point_mask]

    # Resolve either explicit boundaries or evenly spaced boundaries from num_slices.
    boundaries = _resolve_slice_boundaries(
        valid_coordinates_along,
        config.boundaries,
        config.num_slices,
    )

    # Compute slice centres and spans along slicing axis
    spans = np.diff(boundaries)
    centres = 0.5 * (boundaries[:-1] + boundaries[1:])

    # Convert the ROI definition into a geometric object for exact slice intersections.
    roi_geometry = build_roi_geometry(specimen_geometry.region_of_interest.roi_definition)

    # Precompute support lines from the input data (DIC data) rows/columns for width averaging.
    # support positions: mean coordinate along cross-section axis for each row/column
    # support bounds: min and max coordinates along cross-section axis for each row/column
    # support indices: row/column indices corresponding to the support positions and bounds
    support_positions, support_bounds, support_indices = _resolve_cross_section_supports(
        specimen_geometry,
        axis=config.axis,
    )

    # Resolve slice geometry and weighted cross-sections from ROI intersections.
    cross_sections, widths, areas, cross_centres, cross_bounds = _build_slice_geometry_from_roi(
        specimen_geometry,
        roi_geometry=roi_geometry,
        axis=config.axis,
        boundaries=boundaries,
        valid_point_mask=valid_point_mask,
        support_positions=support_positions,
        support_bounds=support_bounds,
        support_indices=support_indices,
    )

    # Associate each valid DIC point with a slice for the constitutive map update.
    slice_id_map, slice_point_indices, point_counts = _associate_points_to_slices(
        specimen_geometry,
        axis=config.axis,
        boundaries=boundaries,
        valid_point_mask=valid_point_mask,
    )

    # Build the slice partition object with all computed properties and associations.
    partition = SlicePartition(
        axis=config.axis,
        boundaries=boundaries,
        centres=centres,
        spans=spans,
        widths=widths,
        areas=areas,
        cross_centres=cross_centres,
        cross_bounds=cross_bounds,
        point_counts=point_counts,
        slice_id_map=slice_id_map,
        valid_point_mask=valid_point_mask,
        slice_point_indices=slice_point_indices,
        cross_sections=cross_sections,
    )

    if plot_diagnostic:
        plot_slice_partition_diagnostic(specimen_geometry, partition)
    return partition


def plot_slice_partition_diagnostic(
    specimen_geometry: SpecimenGeometry,
    slice_partition: SlicePartition,
) -> None:
    """Plot slice rectangles using the average width of each slice."""

    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle

    fig, ax = plt.subplots()
    valid_points = slice_partition.valid_point_mask
    ax.scatter(
        specimen_geometry.x[valid_points],
        specimen_geometry.y[valid_points],
        s=3,
        c="0.75",
        marker="s",
        linewidths=0.0,
        alpha=0.2,
    )

    colours = ("#4C78A8", "#F58518", "#54A24B", "#E45756")
    for slice_index in range(slice_partition.num_slices):
        colour = colours[slice_index % len(colours)]
        if slice_partition.axis == "y":
            rectangle = Rectangle(
                (slice_partition.cross_bounds[slice_index, 0], slice_partition.boundaries[slice_index]),
                slice_partition.widths[slice_index],
                slice_partition.spans[slice_index],
                facecolor=colour,
                edgecolor=colour,
                alpha=0.5,
                linewidth=2,
            )
            text_x = float(slice_partition.cross_centres[slice_index])
            text_y = float(slice_partition.centres[slice_index])
        else:
            rectangle = Rectangle(
                (slice_partition.boundaries[slice_index], slice_partition.cross_bounds[slice_index, 0]),
                slice_partition.spans[slice_index],
                slice_partition.widths[slice_index],
                facecolor=colour,
                edgecolor=colour,
                alpha=0.5,
                linewidth=2,
            )
            text_x = float(slice_partition.centres[slice_index])
            text_y = float(slice_partition.cross_centres[slice_index])

        ax.add_patch(rectangle)
        ax.text(text_x, text_y, str(slice_index), ha="center", va="center", fontsize=9, color="black")

    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title("Slice partition diagnostic (rectangles use average slice width)")
    fig.tight_layout()


def _build_slice_geometry_from_roi(
    specimen_geometry: SpecimenGeometry,
    *,
    roi_geometry: BaseGeometry,
    axis: SliceAxis,
    boundaries: npt.NDArray[np.float64],
    valid_point_mask: npt.NDArray[np.bool_],
    support_positions: npt.NDArray[np.float64],
    support_bounds: npt.NDArray[np.float64],
    support_indices: npt.NDArray[np.int64],
) -> tuple[
    tuple[tuple[SliceCrossSection, ...], ...],
    npt.NDArray[np.float64],
    npt.NDArray[np.float64],
    npt.NDArray[np.float64],
    npt.NDArray[np.float64],
]:
    """Build slice geometry from ROI intersections and native DIC support lines."""

    # Compute the min and max coords across cross-section lines
    cross_field = specimen_geometry.x if axis == "y" else specimen_geometry.y
    cross_values = cross_field[valid_point_mask]
    cross_min = float(np.min(cross_values))
    cross_max = float(np.max(cross_values))

    # Compute small padding that will be added to the cross-section lines to ensure they fully intersect the ROI
    cross_pad = max(0.05 * (cross_max - cross_min), 1.0)

    # Compute slice spans along slicing axis
    spans = np.diff(boundaries)
    
    # Initialize arrays to hold slice properties and cross-section data
    widths = np.zeros(spans.size, dtype=np.float64)
    areas = np.zeros(spans.size, dtype=np.float64)
    cross_centres = np.zeros(spans.size, dtype=np.float64)
    cross_bounds = np.zeros((spans.size, 2), dtype=np.float64)
    slice_cross_sections: list[tuple[SliceCrossSection, ...]] = []

    default_cross_centre = 0.5 * (cross_min + cross_max)
    for slice_index, span in enumerate(spans):
        # Build a rectangular slice band (shapely polygon object) that spans the ROI cross-section
        slice_band =_build_slice_band(
                axis=axis,
                along_min=float(boundaries[slice_index]),
                along_max=float(boundaries[slice_index + 1]),
                cross_min=cross_min,
                cross_max=cross_max,
                cross_pad=cross_pad,
            )
        
        # Compute the polygon of the overlapping region between the ROI and the slice band #NEED TO CHECK THIS WITH HOLE (MULTIPLE POLYGONS?)
        slice_geometry = roi_geometry.intersection(slice_band)


        # Initise accumulators for area and cross-moment calculations
        area_accumulator = 0.0
        cross_moment_accumulator = 0.0
        cross_sections_for_slice: list[SliceCrossSection] = []
        slice_min = float(boundaries[slice_index])
        slice_max = float(boundaries[slice_index + 1])

        # Iterate over each support line and compute the weighted cross-section for the current slice
        for support_idx in range(support_positions.size):
            line_position = float(support_positions[support_idx])
            line_support = support_bounds[support_idx]
            line_index = int(support_indices[support_idx])

            support_min = float(line_support[0])
            support_max = float(line_support[1])
            # Assign weight to the cross-section based on the overlap between the support line and the slice band
            width_weight = _interval_overlap(support_min, support_max, slice_min, slice_max)
            # If the weight is negligible, skip this support line
            if width_weight <= _GEOMETRY_TOLERANCE:
                continue

            cross_section = _build_cross_section(
                specimen_geometry,
                slice_geometry=slice_geometry,
                axis=axis,
                line_position=line_position,
                line_index=line_index,
                cross_min=cross_min,
                cross_max=cross_max,
                cross_pad=cross_pad,
                width_weight=width_weight,
                valid_point_mask=valid_point_mask,
            )
            if cross_section is None:
                continue

            cross_sections_for_slice.append(cross_section)

            weighted_length = cross_section.width_weight * cross_section.line_length
            area_accumulator += weighted_length

            for segment_min, segment_max in cross_section.segment_bounds:
                segment_length = segment_max - segment_min
                segment_centre = 0.5 * (segment_min + segment_max)
                cross_moment_accumulator += (
                    cross_section.width_weight * segment_length * segment_centre
                )

        slice_cross_sections.append(tuple(cross_sections_for_slice))
        areas[slice_index] = area_accumulator
        widths[slice_index] = area_accumulator / span if span > 0.0 else 0.0

        # determine cross-centre from the weighted average of cross-sectional line integrals
        # centroid = (sum of area * centroid) / total area
        if area_accumulator > _GEOMETRY_TOLERANCE:
            cross_centre = cross_moment_accumulator / area_accumulator
        else:
            cross_centre = default_cross_centre

        cross_centres[slice_index] = cross_centre
        cross_bounds[slice_index, 0] = cross_centre - 0.5 * widths[slice_index]
        cross_bounds[slice_index, 1] = cross_centre + 0.5 * widths[slice_index]

    return tuple(slice_cross_sections), widths, areas, cross_centres, cross_bounds


def _build_cross_section(
    specimen_geometry: SpecimenGeometry,
    *,
    slice_geometry: BaseGeometry,
    axis: SliceAxis,
    line_position: float,
    line_index: int,
    cross_min: float,
    cross_max: float,
    cross_pad: float,
    width_weight: float,
    valid_point_mask: npt.NDArray[np.bool_],
) -> SliceCrossSection | None:
    """Build one weighted cross-section line from the geometric ROI intersection."""

    # build a line geometry for the cross-section at the specified position along the slicing axis
    line_geometry = _build_cross_section_line(
        axis=axis,
        line_position=line_position,
        cross_min=cross_min,
        cross_max=cross_max,
        cross_pad=cross_pad,
    )
    # compute segment bounds from the intersection of the slice geometry with the cross-section line
    segment_bounds = _extract_line_segments(slice_geometry.intersection(line_geometry), axis=axis)
    if len(segment_bounds) == 0:
        return None

    # resolve ordered point supports along the cross section line
    point_coordinates, point_cell_bounds, point_indices, point_valid = _resolve_line_point_supports(
        specimen_geometry,
        axis=axis,
        line_index=line_index,
        valid_point_mask=valid_point_mask,
    )

    point_index_list: list[int] = []
    point_coordinate_list: list[float] = []
    point_cell_bound_list: list[tuple[float, float]] = []
    point_weight_list: list[float] = []
    point_segment_id_list: list[int] = []

    # Iterate over each segment in the cross-section and associate valid DIC points with the segment
    for segment_id, (segment_min, segment_max) in enumerate(segment_bounds):
        segment_has_valid_point = False
        for point_idx in range(point_indices.size):
            point_index = int(point_indices[point_idx])
            point_coordinate = float(point_coordinates[point_idx])
            point_cell_min = float(point_cell_bounds[point_idx, 0])
            point_cell_max = float(point_cell_bounds[point_idx, 1])
            is_valid = bool(point_valid[point_idx])

            overlap_length = _interval_overlap(
                point_cell_min,
                point_cell_max,
                segment_min,
                segment_max,
            )
            if overlap_length <= _GEOMETRY_TOLERANCE or not is_valid:
                continue

            point_index_list.append(point_index)
            point_coordinate_list.append(point_coordinate)
            point_cell_bound_list.append((point_cell_min, point_cell_max))
            point_weight_list.append(overlap_length)
            point_segment_id_list.append(segment_id)
            segment_has_valid_point = True

        if not segment_has_valid_point and (segment_max - segment_min) > _GEOMETRY_TOLERANCE:
            raise ValueError(
                "A slice cross-section intersects the ROI but no valid DIC points were available on that "
                "material segment. Refine the slice definition or check the ROI/data alignment."
            )

    if len(point_index_list) == 0:
        return None

    return SliceCrossSection(
        position=line_position,
        width_weight=width_weight,
        line_length=float(sum(segment_max - segment_min for segment_min, segment_max in segment_bounds)),
        segment_bounds=segment_bounds,
        point_coordinates=np.asarray(point_coordinate_list, dtype=np.float64),
        point_cell_bounds=np.asarray(point_cell_bound_list, dtype=np.float64),
        point_indices=np.asarray(point_index_list, dtype=np.int64),
        point_area_integral_weights=np.asarray(point_weight_list, dtype=np.float64),
        point_segment_ids=np.asarray(point_segment_id_list, dtype=np.int32),
    )


def _associate_points_to_slices(
    specimen_geometry: SpecimenGeometry,
    *,
    axis: SliceAxis,
    boundaries: npt.NDArray[np.float64],
    valid_point_mask: npt.NDArray[np.bool_],
) -> tuple[npt.NDArray[np.int32], tuple[npt.NDArray[np.int64], ...], npt.NDArray[np.int64]]:
    """Associate each valid DIC point with a slice for constitutive updates."""

    coordinate_along = specimen_geometry.x if axis == "x" else specimen_geometry.y
    valid_coordinates_along = coordinate_along[valid_point_mask]
    # Use searchsorted to find the slice index for each valid coordinate along the slicing axis
    slice_indices = np.searchsorted(boundaries, valid_coordinates_along, side="right") - 1
    slice_indices = np.clip(slice_indices, 0, boundaries.size - 2)

    if np.any(valid_coordinates_along < boundaries[0]) or np.any(valid_coordinates_along > boundaries[-1]):
        raise ValueError("Slice boundaries do not fully cover the valid ROI extent.")

    # Build a map of slice indices for all points in the specimen geometry, initializing with -1 for invalid points
    slice_id_map = np.full(specimen_geometry.x.shape, -1, dtype=np.int32)
    valid_flat_indices = np.flatnonzero(valid_point_mask)
    slice_id_map.ravel()[valid_flat_indices] = slice_indices.astype(np.int32)

    # Count the number of valid points in each slice and collect their indices
    point_counts = np.zeros(boundaries.size - 1, dtype=np.int64)
    slice_point_indices: list[npt.NDArray[np.int64]] = []
    for slice_index in range(boundaries.size - 1):
        flat_indices = valid_flat_indices[slice_indices == slice_index].astype(np.int64, copy=False)
        slice_point_indices.append(flat_indices)
        point_counts[slice_index] = flat_indices.size

    return slice_id_map, tuple(slice_point_indices), point_counts


def _resolve_cross_section_supports(
    specimen_geometry: SpecimenGeometry,
    *,
    axis: SliceAxis,
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64], npt.NDArray[np.int64]]:
    """Resolve the native DIC row/column centres and their support intervals.
       The support intervals are used to compute the average slice width from the
       weighted average of cross-sectional line integrals.
    """

    if axis == "y":
        # Compute the mean y-coordinate for each row (axis=1) to get the support coordinates.
        support_coordinates = _nanmean_without_warnings(specimen_geometry.y, axis=1)
    else:
        # Compute the mean x-coordinate for each column (axis=0) to get the support coordinates.
        support_coordinates = _nanmean_without_warnings(specimen_geometry.x, axis=0)

    valid_support_indices = np.flatnonzero(np.isfinite(support_coordinates))
    positions = support_coordinates[valid_support_indices]
    order = np.argsort(positions)
    ordered_positions = positions[order]
    ordered_indices = valid_support_indices[order].astype(np.int64, copy=False)
    support_bounds = _compute_support_bounds(ordered_positions)
    return ordered_positions, support_bounds, ordered_indices


def _resolve_line_point_supports(
    specimen_geometry: SpecimenGeometry,
    *,
    axis: SliceAxis,
    line_index: int,
    valid_point_mask: npt.NDArray[np.bool_],
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64], npt.NDArray[np.int64], npt.NDArray[np.bool_]]:
    """Resolve ordered point supports along one row or column.
    
       Note: This currently assumes slices are defined along the x or y axes 
       and that the DIC data is structured as a regular grid. So angled slices aren't supported yet.
    """

    if axis == "y":
        point_coordinates_raw = specimen_geometry.x[line_index, :]
        point_valid_raw = valid_point_mask[line_index, :]
        row_indices = np.full(point_coordinates_raw.shape, line_index, dtype=np.int64)
        col_indices = np.arange(point_coordinates_raw.size, dtype=np.int64)
    else:
        point_coordinates_raw = specimen_geometry.y[:, line_index]
        point_valid_raw = valid_point_mask[:, line_index]
        row_indices = np.arange(point_coordinates_raw.size, dtype=np.int64)
        col_indices = np.full(point_coordinates_raw.shape, line_index, dtype=np.int64)

    finite_points = np.isfinite(point_coordinates_raw)
    point_coordinates = point_coordinates_raw[finite_points]
    point_valid = point_valid_raw[finite_points]
    row_indices = row_indices[finite_points]
    col_indices = col_indices[finite_points]

    order = np.argsort(point_coordinates)
    ordered_coordinates = point_coordinates[order].astype(np.float64, copy=False)
    ordered_valid = point_valid[order].astype(bool, copy=False)
    ordered_rows = row_indices[order]
    ordered_cols = col_indices[order]
    point_indices = np.ravel_multi_index((ordered_rows, ordered_cols), specimen_geometry.x.shape).astype(np.int64)
    point_cell_bounds = _compute_support_bounds(ordered_coordinates)
    return ordered_coordinates, point_cell_bounds, point_indices, ordered_valid


def _build_slice_band(
    *,
    axis: SliceAxis,
    along_min: float,
    along_max: float,
    cross_min: float,
    cross_max: float,
    cross_pad: float,
) -> Polygon:
    """Build a rectangular slice band spanning the ROI in the cross direction.
    
       Shapely box has definition: box(minx, miny, maxx, maxy, ccw=True)
    """

    if axis == "y":
        min_x, max_x = cross_min - cross_pad, cross_max + cross_pad
        min_y, max_y = along_min, along_max
    else:
        min_x, max_x = along_min, along_max
        min_y, max_y = cross_min - cross_pad, cross_max + cross_pad

    return box(min_x, min_y, max_x, max_y)


def _build_cross_section_line(
    *,
    axis: SliceAxis,
    line_position: float,
    cross_min: float,
    cross_max: float,
    cross_pad: float,
) -> LineString:
    """Build a horizontal or vertical cross-section line through the ROI."""

    if axis == "y":
        return LineString(
            (
                (cross_min - cross_pad, line_position),
                (cross_max + cross_pad, line_position),
            )
        )
    return LineString(
        (
            (line_position, cross_min - cross_pad),
            (line_position, cross_max + cross_pad),
        )
    )


def _extract_line_segments(
    geometry: BaseGeometry,
    *,
    axis: SliceAxis,
) -> tuple[tuple[float, float], ...]:
    """Extract sorted line segments from a line/ROI intersection result."""

    raw_segments: list[tuple[float, float]] = []

    def append_segments(current_geometry: BaseGeometry) -> None:
        if current_geometry.is_empty:
            return
        if isinstance(current_geometry, LineString):
            coordinates = np.asarray(current_geometry.coords, dtype=np.float64)
            if coordinates.shape[0] < 2:
                return
            line_coordinates = coordinates[:, 0] if axis == "y" else coordinates[:, 1]
            segment_min = float(np.min(line_coordinates))
            segment_max = float(np.max(line_coordinates))
            if segment_max - segment_min > _GEOMETRY_TOLERANCE:
                raw_segments.append((segment_min, segment_max))
            return
        if isinstance(current_geometry, MultiLineString):
            for line in current_geometry.geoms:
                append_segments(line)
            return
        if isinstance(current_geometry, GeometryCollection):
            for item in current_geometry.geoms:
                append_segments(item)

    append_segments(geometry)
    if len(raw_segments) == 0:
        return tuple()

    raw_segments.sort(key=lambda segment: segment[0])
    merged_segments: list[tuple[float, float]] = [raw_segments[0]]
    for segment_min, segment_max in raw_segments[1:]:
        current_min, current_max = merged_segments[-1]
        if segment_min <= current_max + _GEOMETRY_TOLERANCE:
            merged_segments[-1] = (current_min, max(current_max, segment_max))
        else:
            merged_segments.append((segment_min, segment_max))
    return tuple(merged_segments)


def _compute_support_bounds(
    coordinates: npt.NDArray[np.float64],
) -> npt.NDArray[np.float64]:
    """Compute piecewise-constant support cells around ordered point centres."""

    if coordinates.ndim != 1 or coordinates.size == 0:
        raise ValueError("Support coordinates must be a non-empty 1D array.")
    if coordinates.size == 1:
        half_width = 0.5
        return np.asarray([[coordinates[0] - half_width, coordinates[0] + half_width]], dtype=np.float64)

    midpoints = 0.5 * (coordinates[:-1] + coordinates[1:])
    left_edge = coordinates[0] - 0.5 * (coordinates[1] - coordinates[0])
    right_edge = coordinates[-1] + 0.5 * (coordinates[-1] - coordinates[-2])
    boundaries = np.concatenate(([left_edge], midpoints, [right_edge]))
    return np.column_stack((boundaries[:-1], boundaries[1:]))


def _nanmean_without_warnings(
    values: npt.NDArray[np.float64],
    *,
    axis: int,
) -> npt.NDArray[np.float64]:
    """Compute nanmean while returning NaN for all-NaN rows/columns."""

    finite_mask = np.isfinite(values)
    sums = np.sum(np.where(finite_mask, values, 0.0), axis=axis)
    counts = np.sum(finite_mask, axis=axis)
    means = np.full(sums.shape, np.nan, dtype=np.float64)
    np.divide(sums, counts, out=means, where=counts > 0)
    return means


def _interval_overlap(
    interval_min: float,
    interval_max: float,
    target_min: float,
    target_max: float,
) -> float:
    """Return the overlap length between two 1D intervals.
    
       interal_min, interval_max: The min and max of the first interval.
       target_min, target_max: The min and max of the second interval.
       left edge is later of the two starts: min(interval_max, target_max)
       right edge is earlier of the two ends: max(interval_min, target_min)
       
       """

    return max(0.0, min(interval_max, target_max) - max(interval_min, target_min))


@dataclass(slots=True)
class SliceWiseSpatialParameterisation(ISpatialParameterisation):
    slice_partition: SlicePartition
    values: list[float | DegreeOfFreedom | None] | None = None

    def __post_init__(self) -> None:
        if self.values is None:
            self.values = [None] * self.slice_partition.num_slices
        if len(self.values) != self.slice_partition.num_slices:
            raise ValueError(
                f"Expected {self.slice_partition.num_slices} slice values, got {len(self.values)}."
            )

    def get_num_degrees_of_freedom(self) -> int:
        return sum(isinstance(value, DegreeOfFreedom) or value is None for value in self.values)

    def update_from_constitutive_parameter(
        self,
        constitutive_parameter: ConstitutiveParameter,
    ) -> None:
        parameter_values = np.asarray(constitutive_parameter.value, dtype=np.float64)
        if parameter_values.shape != self.slice_partition.slice_id_map.shape:
            raise ValueError(
                "Constitutive parameter map shape does not match the slice partition shape: "
                f"{parameter_values.shape} vs {self.slice_partition.slice_id_map.shape}."
            )

        global_mean = float(np.nanmean(parameter_values))
        updated_values: list[float | DegreeOfFreedom] = []
        for slice_index, current_value in enumerate(self.values):
            slice_mask = self.slice_partition.slice_id_map == slice_index
            finite_slice_values = parameter_values[slice_mask & np.isfinite(parameter_values)]
            slice_mean = float(np.mean(finite_slice_values)) if finite_slice_values.size > 0 else global_mean

            if isinstance(current_value, DegreeOfFreedom):
                updated_values.append(
                    DegreeOfFreedom(
                        slice_mean,
                        current_value.lower_bound,
                        current_value.upper_bound,
                    )
                )
            elif current_value is None:
                updated_values.append(
                    DegreeOfFreedom(
                        slice_mean,
                        constitutive_parameter.lower_bound,
                        constitutive_parameter.upper_bound,
                    )
                )
            else:
                updated_values.append(slice_mean)

        self.values = updated_values

    def to_map(
        self,
        size: npt.NDArray[np.uint32],
    ) -> npt.NDArray[np.float64]:
        map_shape = (int(size[0]), int(size[1]))
        if map_shape != self.slice_partition.slice_id_map.shape:
            raise ValueError(
                f"Requested map shape {map_shape} does not match the slice partition shape "
                f"{self.slice_partition.slice_id_map.shape}."
            )

        slice_values = np.asarray([_resolve_slice_value(value) for value in self.values], dtype=np.float64)
        parameter_map = np.full(map_shape, float(np.mean(slice_values)), dtype=np.float64)
        for slice_index, slice_value in enumerate(slice_values):
            parameter_map[self.slice_partition.slice_id_map == slice_index] = slice_value
        return parameter_map

    def collect_degrees_of_freedom(self) -> list[DegreeOfFreedom]:
        dofs: list[DegreeOfFreedom] = []
        for value in self.values:
            if isinstance(value, DegreeOfFreedom):
                dofs.append(copy(value))
            elif value is None:
                raise ValueError(
                    "SliceWiseSpatialParameterisation must be initialised with "
                    "update_from_constitutive_parameter before collecting DOFs."
                )
        return dofs

    def update_from_degrees_of_freedom(
        self,
        degrees_of_freedom: list[DegreeOfFreedom] | npt.NDArray[np.float64],
    ) -> None:
        dof_index = 0
        for slice_index, value in enumerate(self.values):
            if not isinstance(value, DegreeOfFreedom):
                continue

            updated_value = degrees_of_freedom[dof_index]
            if isinstance(updated_value, DegreeOfFreedom):
                self.values[slice_index] = updated_value
            else:
                value.value = float(updated_value)
            dof_index += 1


def _resolve_slice_value(value: float | DegreeOfFreedom | None) -> float:
    if isinstance(value, DegreeOfFreedom):
        return float(value.value)
    if value is None:
        raise ValueError(
            "SliceWiseSpatialParameterisation must be initialised with "
            "update_from_constitutive_parameter before converting to a map."
        )
    return float(value)


def _resolve_slice_boundaries(
    valid_coordinates_along: npt.NDArray[np.float64],
    boundaries: npt.NDArray[np.float64] | None,
    num_slices: int | None,
) -> npt.NDArray[np.float64]:
    if boundaries is not None:
        resolved = np.asarray(boundaries, dtype=np.float64)
        if resolved.ndim != 1 or resolved.size < 2:
            raise ValueError("Slice boundaries must be a 1D array with at least two entries.")
        if not np.all(np.diff(resolved) > 0.0):
            raise ValueError("Slice boundaries must be strictly increasing.")
    else:
        if num_slices is None:
            raise ValueError("num_slices must be provided when boundaries is not provided.")
        if num_slices < 1:
            raise ValueError("num_slices must be at least 1.")
        resolved = np.linspace(
            float(np.min(valid_coordinates_along)),
            float(np.max(valid_coordinates_along)),
            num_slices + 1,
        )

    if np.min(valid_coordinates_along) < resolved[0] or np.max(valid_coordinates_along) > resolved[-1]:
        raise ValueError("Slice boundaries do not fully cover the valid ROI extent.")
    if np.any(np.diff(resolved) <= 0.0):
        raise ValueError("Slice boundaries produced zero-width or negative-width slices.")
    return resolved


# Future option if we want to re-enable automatic axis selection.
# def infer_loading_axis(edge_conditions: EdgeConditions) -> SliceAxis:
#     x_has_traction = (
#         edge_conditions.min_x_edge.x is EEdgeCondition.Traction
#         or edge_conditions.max_x_edge.x is EEdgeCondition.Traction
#     )
#     y_has_traction = (
#         edge_conditions.min_y_edge.y is EEdgeCondition.Traction
#         or edge_conditions.max_y_edge.y is EEdgeCondition.Traction
#     )
#
#     if x_has_traction and y_has_traction:
#         raise ValueError("Could not infer a unique loading axis because both x and y edges have traction DOFs.")
#     if x_has_traction:
#         return "x"
#     if y_has_traction:
#         return "y"
#     raise ValueError("Could not infer the loading axis from edge conditions. Please set it explicitly.")


SlicewiseSpatialParameterisation = SliceWiseSpatialParameterisation
