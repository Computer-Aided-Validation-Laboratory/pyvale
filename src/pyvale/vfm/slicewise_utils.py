from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Protocol

import numpy as np
import numpy.typing as npt
from shapely.geometry import GeometryCollection, MultiPolygon, Polygon, box
from shapely.geometry.base import BaseGeometry

from pyvale.vfm.experimentdata import SpecimenGeometry
from pyvale.vfm.roi import VfmRegionOfInterest, build_roi_geometry
from pyvale.vfm.vfmesh import _generate_data_mesh_nodal_coord


SliceAxis = Literal["x", "y"]
_GEOMETRY_TOLERANCE = 1.0e-9


@dataclass(slots=True, frozen=True)
class SliceConfig:
    """Configuration for slice-wise partitioning.

    Axis must be provided explicitly. Provide either `num_slices` or
    `boundaries`. These values define the intended geometric slices; the
    resolved slice partition snaps internal boundaries to support-cell edges
    so datapoint cells remain the smallest material unit.
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


class SliceAssignmentPartition(Protocol):
    """Minimal partition interface required by slice-wise parameter maps."""

    num_slices: int
    slice_id_map: npt.NDArray[np.int32]


def _build_valid_point_mask(
    specimen_geometry: SpecimenGeometry,
) -> npt.NDArray[np.bool_]:
    specimen_mask = specimen_geometry.region_of_interest.sample_specimen_mask(
        specimen_geometry.x,
        specimen_geometry.y,
    )
    return (
        specimen_mask
        & np.isfinite(specimen_geometry.x)
        & np.isfinite(specimen_geometry.y)
        & np.isfinite(specimen_geometry.pixel_area)
    )


def _associate_points_to_slices(
    specimen_geometry: SpecimenGeometry,
    *,
    axis: SliceAxis,
    boundaries: npt.NDArray[np.float64],
    valid_point_mask: npt.NDArray[np.bool_],
) -> tuple[npt.NDArray[np.int32], tuple[npt.NDArray[np.int64], ...], npt.NDArray[np.int64]]:
    """Associate each valid DIC point with the slice containing its centre."""

    coordinate_along = specimen_geometry.x if axis == "x" else specimen_geometry.y
    valid_coordinates_along = coordinate_along[valid_point_mask]
    slice_indices = np.searchsorted(boundaries, valid_coordinates_along, side="right") - 1
    slice_indices = np.clip(slice_indices, 0, boundaries.size - 2)

    if np.any(valid_coordinates_along < boundaries[0]) or np.any(valid_coordinates_along > boundaries[-1]):
        raise ValueError("Slice boundaries do not fully cover the valid ROI extent.")

    slice_id_map = np.full(specimen_geometry.x.shape, -1, dtype=np.int32)
    valid_flat_indices = np.flatnonzero(valid_point_mask)
    slice_id_map.ravel()[valid_flat_indices] = slice_indices.astype(np.int32)

    point_counts = np.zeros(boundaries.size - 1, dtype=np.int64)
    slice_point_indices: list[npt.NDArray[np.int64]] = []
    for slice_index in range(boundaries.size - 1):
        flat_indices = valid_flat_indices[slice_indices == slice_index].astype(np.int64, copy=False)
        slice_point_indices.append(flat_indices)
        point_counts[slice_index] = flat_indices.size

    return slice_id_map, tuple(slice_point_indices), point_counts


def _build_slice_band(
    *,
    axis: SliceAxis,
    along_min: float,
    along_max: float,
    cross_min: float,
    cross_max: float,
    cross_pad: float,
):
    """Build a padded rectangular band that spans one slice along the chosen axis."""

    if axis == "y":
        return box(cross_min - cross_pad, along_min, cross_max + cross_pad, along_max)
    return box(along_min, cross_min - cross_pad, along_max, cross_max + cross_pad)


@dataclass(slots=True, frozen=True)
class SliceAreaPartition:
    """Area-based slice partition backed by DIC support-cell overlap areas.

    `areas` and `widths` are the discrete operator values implied by the
    support-cell overlaps. `geometric_areas` and `geometric_widths` are the
    exact ROI/slice polygon values retained for diagnostics.
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
    slice_force_point_area_integral_weights: tuple[npt.NDArray[np.float64], ...]
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


def _resolve_slice_boundaries(
    valid_coordinates_along: npt.NDArray[np.float64],
    boundaries: npt.NDArray[np.float64] | None,
    num_slices: int | None,
) -> npt.NDArray[np.float64]:
    """Resolve explicit or evenly spaced slice boundaries along one axis."""

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


def resolve_cell_aligned_slice_boundaries(
    specimen_geometry: SpecimenGeometry,
    slice_config: SliceConfig,
) -> npt.NDArray[np.float64]:
    """Resolve slices, then snap internal boundaries to support-cell edges.

    The VFM slice-wise material maps treat each DIC datapoint cell as the
    smallest material representative volume. Snapping internal slice
    boundaries to support-cell edges keeps that material discretisation
    consistent with the area-overlap force operator while preserving the
    requested geometric slice layout as closely as the data resolution allows.
    """

    valid_point_mask = _build_valid_point_mask(specimen_geometry)
    if not np.any(valid_point_mask):
        raise ValueError("No valid specimen points were available to align slice boundaries.")

    coordinate_along = specimen_geometry.x if slice_config.axis == "x" else specimen_geometry.y
    valid_coordinates_along = coordinate_along[valid_point_mask]
    intended_boundaries = _resolve_slice_boundaries(
        valid_coordinates_along,
        slice_config.boundaries,
        slice_config.num_slices,
    )
    if intended_boundaries.size <= 2:
        return intended_boundaries.copy()

    support_node_x, support_node_y = _build_support_node_grids(specimen_geometry)
    candidate_edges = _collect_support_cell_edges_along(
        specimen_geometry,
        axis=slice_config.axis,
        valid_point_mask=valid_point_mask,
        support_node_x=support_node_x,
        support_node_y=support_node_y,
        lower_bound=float(intended_boundaries[0]),
        upper_bound=float(intended_boundaries[-1]),
    )
    num_internal_boundaries = intended_boundaries.size - 2
    if candidate_edges.size < num_internal_boundaries:
        raise ValueError(
            "Cannot align the requested slice boundaries to support-cell edges: "
            f"{num_internal_boundaries} internal boundaries were requested, but only "
            f"{candidate_edges.size} candidate cell edges are available inside the ROI extent."
        )

    snapped_internal = _snap_internal_boundaries_monotonic(
        intended_boundaries[1:-1],
        candidate_edges,
    )
    resolved = np.concatenate(
        (
            intended_boundaries[:1],
            snapped_internal,
            intended_boundaries[-1:],
        )
    )
    if np.any(np.diff(resolved) <= 0.0):
        raise ValueError("Cell-aligned slice boundaries are not strictly increasing.")
    return resolved


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

    valid_point_mask = _build_valid_point_mask(specimen_geometry)
    if not np.any(valid_point_mask):
        raise ValueError("No valid specimen points were available to build the area-based slice partition.")

    boundaries = resolve_cell_aligned_slice_boundaries(
        specimen_geometry,
        config,
    )

    spans = np.diff(boundaries)
    centres = 0.5 * (boundaries[:-1] + boundaries[1:])
    roi_geometry = build_roi_geometry(specimen_geometry.region_of_interest.roi_definition)
    slice_geometries, geometric_areas = _build_slice_geometries(
        roi_geometry=roi_geometry,
        axis=config.axis,
        boundaries=boundaries,
    )

    geometric_widths = np.divide(
        geometric_areas,
        spans,
        out=np.zeros_like(geometric_areas),
        where=spans > 0.0,
    )

    support_node_x, support_node_y = _build_support_node_grids(specimen_geometry)
    (
        slice_force_point_indices,
        slice_force_point_areas,
        slice_force_point_area_integral_weights,
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

    widths = np.divide(
        discrete_areas,
        spans,
        out=np.zeros_like(discrete_areas),
        where=spans > 0.0,
    )
    coverage_fractions = np.divide(
        discrete_areas,
        geometric_areas,
        out=np.ones_like(discrete_areas),
        where=geometric_areas > _GEOMETRY_TOLERANCE,
    )

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
        slice_force_point_area_integral_weights=slice_force_point_area_integral_weights,
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

    if slice_index is not None and (slice_index < 0 or slice_index >= slice_partition.num_slices):
        raise IndexError(f"Slice index {slice_index} is out of range.")

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


def calculate_roi_slice_areas(
    region_of_interest: VfmRegionOfInterest,
    *,
    axis: SliceAxis,
    boundaries: npt.NDArray[np.float64],
) -> npt.NDArray[np.float64]:
    """Return exact ROI area intersecting each longitudinal slice.

    This public helper is used by the optional FRE-only physical-domain
    correction.  Slice boundaries still come from the measured DIC support;
    only the cross-sectional area comes from the supplied physical outline.
    """

    roi_geometry = build_roi_geometry(region_of_interest.roi_definition)
    _, areas = _build_slice_geometries(
        roi_geometry=roi_geometry,
        axis=axis,
        boundaries=np.asarray(boundaries, dtype=np.float64),
    )
    return areas


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
    for row_index, col_index in zip(valid_rows, valid_cols, strict=True):
        row_index = int(row_index)
        col_index = int(col_index)

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

        along_coordinates = cell_vertices[:, 0] if axis == "x" else cell_vertices[:, 1]
        along_min = float(np.min(along_coordinates))
        along_max = float(np.max(along_coordinates))

        slice_start = int(np.searchsorted(boundaries, along_min, side="right") - 1)
        slice_end = int(np.searchsorted(boundaries, along_max, side="left") - 1)
        if slice_end < 0 or slice_start > num_slices - 1:
            continue

        slice_start = max(0, slice_start)
        slice_end = min(num_slices - 1, slice_end)
        flat_index = int(np.ravel_multi_index((row_index, col_index), specimen_geometry.x.shape))

        for slice_index in range(slice_start, slice_end + 1):
            if _interval_overlap(
                along_min,
                along_max,
                float(boundaries[slice_index]),
                float(boundaries[slice_index + 1]),
            ) <= _GEOMETRY_TOLERANCE:
                continue

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


def _collect_support_cell_edges_along(
    specimen_geometry: SpecimenGeometry,
    *,
    axis: SliceAxis,
    valid_point_mask: npt.NDArray[np.bool_],
    support_node_x: npt.NDArray[np.float64],
    support_node_y: npt.NDArray[np.float64],
    lower_bound: float,
    upper_bound: float,
) -> npt.NDArray[np.float64]:
    """Return sorted support-cell edge coordinates inside the slice extent."""

    edge_coordinates: list[float] = []
    valid_rows, valid_cols = np.nonzero(valid_point_mask)
    for row_index, col_index in zip(valid_rows, valid_cols, strict=True):
        vertices = _build_support_cell_vertices(
            support_node_x,
            support_node_y,
            int(row_index),
            int(col_index),
        )
        along = vertices[:, 0] if axis == "x" else vertices[:, 1]
        edge_coordinates.extend((float(np.min(along)), float(np.max(along))))

    if not edge_coordinates:
        return np.zeros(0, dtype=np.float64)

    edges = np.asarray(edge_coordinates, dtype=np.float64)
    edges = edges[np.isfinite(edges)]
    edge_tolerance = _estimate_edge_tolerance(edges)
    internal_edges = edges[
        (edges > lower_bound + edge_tolerance)
        & (edges < upper_bound - edge_tolerance)
    ]
    if internal_edges.size == 0:
        return np.zeros(0, dtype=np.float64)
    rounded_edges = np.round(internal_edges / edge_tolerance) * edge_tolerance
    return np.unique(rounded_edges)


def _estimate_edge_tolerance(
    values: npt.NDArray[np.float64],
) -> float:
    finite_values = np.sort(np.asarray(values, dtype=np.float64)[np.isfinite(values)])
    if finite_values.size < 2:
        return 1.0e-12
    spacings = np.diff(np.unique(finite_values))
    positive_spacings = spacings[spacings > 0.0]
    if positive_spacings.size == 0:
        return 1.0e-12
    return max(1.0e-12, 1.0e-8 * float(np.min(positive_spacings)))


def _snap_internal_boundaries_monotonic(
    intended_internal_boundaries: npt.NDArray[np.float64],
    candidate_edges: npt.NDArray[np.float64],
) -> npt.NDArray[np.float64]:
    """Snap boundaries to unique candidate edges while preserving ordering."""

    intended = np.asarray(intended_internal_boundaries, dtype=np.float64)
    candidates = np.asarray(candidate_edges, dtype=np.float64)
    if candidates.size < intended.size:
        raise ValueError("Not enough candidate support-cell edges for the requested slice boundaries.")

    snapped = np.empty(intended.shape, dtype=np.float64)
    first_allowed_index = 0
    for boundary_index, intended_boundary in enumerate(intended):
        remaining_boundaries = intended.size - boundary_index - 1
        last_allowed_index = candidates.size - remaining_boundaries
        candidate_window = candidates[first_allowed_index:last_allowed_index]
        if candidate_window.size == 0:
            raise ValueError("Could not find an ordered support-cell edge for every slice boundary.")
        local_index = int(np.argmin(np.abs(candidate_window - intended_boundary)))
        selected_index = first_allowed_index + local_index
        snapped[boundary_index] = candidates[selected_index]
        first_allowed_index = selected_index + 1

    return snapped


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


def _interval_overlap(
    interval_min: float,
    interval_max: float,
    target_min: float,
    target_max: float,
) -> float:
    return max(0.0, min(interval_max, target_max) - max(interval_min, target_min))


def slice_partition_matches_config(
    slice_partition: SliceAreaPartition,
    slice_config: SliceConfig,
    specimen_geometry: SpecimenGeometry | None = None,
) -> bool:
    if slice_partition.axis != slice_config.axis:
        return False

    if slice_config.boundaries is not None:
        if specimen_geometry is None:
            return np.allclose(slice_partition.boundaries, slice_config.boundaries)
        resolved_boundaries = resolve_cell_aligned_slice_boundaries(
            specimen_geometry,
            slice_config,
        )
        return np.allclose(slice_partition.boundaries, resolved_boundaries)

    return slice_partition.num_slices == slice_config.num_slices


def slice_partitions_are_equivalent(
    lhs: SliceAreaPartition,
    rhs: SliceAreaPartition,
) -> bool:
    return (
        lhs.axis == rhs.axis
        and lhs.num_slices == rhs.num_slices
        and lhs.slice_id_map.shape == rhs.slice_id_map.shape
        and np.allclose(lhs.boundaries, rhs.boundaries)
        and np.array_equal(lhs.slice_id_map, rhs.slice_id_map)
    )


def resolve_slice_partition(
    specimen_geometry: SpecimenGeometry,
    *,
    slice_partition: SliceAreaPartition | None,
    slice_config: SliceConfig | None,
    plot_diagnostic: bool = False,
    diagnostic_slice_index: int | None = None,
) -> SliceAreaPartition:
    """Resolve a usable partition from either an existing partition or a config."""

    if slice_partition is not None:
        if slice_config is not None and not slice_partition_matches_config(
            slice_partition,
            slice_config,
            specimen_geometry,
        ):
            raise ValueError("Provided slice_partition is inconsistent with the provided slice_config.")
        return slice_partition

    if slice_config is None:
        raise ValueError("Provide either slice_partition or slice_config.")

    return build_slice_area_partition(
        specimen_geometry,
        slice_config=slice_config,
        plot_diagnostic=plot_diagnostic,
        diagnostic_slice_index=diagnostic_slice_index,
    )


SlicePartition = SliceAreaPartition
build_slice_partition = build_slice_area_partition
