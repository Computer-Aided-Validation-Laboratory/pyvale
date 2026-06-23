from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
import numpy.typing as npt

from pyvale.vfm.experimentdata import EdgeConditions, EEdgeCondition, SpecimenGeometry


SliceAxis = Literal["x", "y"]


@dataclass(slots=True, frozen=True)
class SliceConfig:
    axis: SliceAxis | None = None
    num_slices: int = 10
    boundaries: npt.NDArray[np.float64] | None = None


@dataclass(slots=True, frozen=True)
class SlicePartition:
    axis: SliceAxis
    boundaries: npt.NDArray[np.float64]
    centres: npt.NDArray[np.float64]
    spans: npt.NDArray[np.float64]
    widths: npt.NDArray[np.float64]
    areas: npt.NDArray[np.float64]
    point_counts: npt.NDArray[np.int64]
    slice_id_map: npt.NDArray[np.int32]
    valid_point_mask: npt.NDArray[np.bool_]
    slice_point_indices: tuple[npt.NDArray[np.int64], ...]

    @property
    def num_slices(self) -> int:
        return int(self.spans.size)

    def get_slice_mask(self, slice_index: int) -> npt.NDArray[np.bool_]:
        return self.slice_id_map == slice_index


def infer_loading_axis(edge_conditions: EdgeConditions) -> SliceAxis:
    x_has_traction = (
        edge_conditions.min_x_edge.x is EEdgeCondition.Traction
        or edge_conditions.max_x_edge.x is EEdgeCondition.Traction
    )
    y_has_traction = (
        edge_conditions.min_y_edge.y is EEdgeCondition.Traction
        or edge_conditions.max_y_edge.y is EEdgeCondition.Traction
    )

    if x_has_traction and y_has_traction:
        raise ValueError("Could not infer a unique loading axis because both x and y edges have traction DOFs.")
    if x_has_traction:
        return "x"
    if y_has_traction:
        return "y"
    raise ValueError("Could not infer the loading axis from edge conditions. Please set it explicitly.")


def build_slice_partition(
    specimen_geometry: SpecimenGeometry,
    *,
    edge_conditions: EdgeConditions | None = None,
    slice_config: SliceConfig | None = None,
) -> SlicePartition:
    config = SliceConfig() if slice_config is None else slice_config
    if config.axis is None:
        if edge_conditions is None:
            raise ValueError("slice axis was not provided and could not be inferred without edge_conditions.")
        axis = infer_loading_axis(edge_conditions)
    else:
        axis = config.axis

    valid_point_mask = (
        specimen_geometry.specimen_mask
        & np.isfinite(specimen_geometry.x)
        & np.isfinite(specimen_geometry.y)
        & np.isfinite(specimen_geometry.pixel_area)
    )
    if not np.any(valid_point_mask):
        raise ValueError("No valid specimen points were available to build the slice partition.")

    coordinate_along = specimen_geometry.x if axis == "x" else specimen_geometry.y
    valid_coordinates_along = coordinate_along[valid_point_mask]

    boundaries = _resolve_slice_boundaries(
        valid_coordinates_along,
        config.boundaries,
        config.num_slices,
    )
    spans = np.diff(boundaries)
    centres = 0.5 * (boundaries[:-1] + boundaries[1:])

    slice_indices = np.searchsorted(boundaries, valid_coordinates_along, side="right") - 1
    slice_indices = np.clip(slice_indices, 0, spans.size - 1)

    if np.any(valid_coordinates_along < boundaries[0]) or np.any(valid_coordinates_along > boundaries[-1]):
        raise ValueError("Slice boundaries do not fully cover the valid ROI extent.")

    slice_id_map = np.full(specimen_geometry.x.shape, -1, dtype=np.int32)
    valid_flat_indices = np.flatnonzero(valid_point_mask)
    slice_id_map.ravel()[valid_flat_indices] = slice_indices.astype(np.int32)

    widths = np.zeros(spans.size, dtype=np.float64)
    areas = np.zeros(spans.size, dtype=np.float64)
    point_counts = np.zeros(spans.size, dtype=np.int64)
    slice_point_indices: list[npt.NDArray[np.int64]] = []

    pixel_area_flat = specimen_geometry.pixel_area.ravel()
    for slice_index in range(spans.size):
        flat_indices = valid_flat_indices[slice_indices == slice_index]
        slice_point_indices.append(flat_indices.astype(np.int64, copy=False))
        point_counts[slice_index] = flat_indices.size
        if flat_indices.size == 0:
            continue
        areas[slice_index] = float(np.nansum(pixel_area_flat[flat_indices]))
        widths[slice_index] = areas[slice_index] / spans[slice_index]

    return SlicePartition(
        axis=axis,
        boundaries=boundaries,
        centres=centres,
        spans=spans,
        widths=widths,
        areas=areas,
        point_counts=point_counts,
        slice_id_map=slice_id_map,
        valid_point_mask=valid_point_mask,
        slice_point_indices=tuple(slice_point_indices),
    )


def _resolve_slice_boundaries(
    valid_coordinates_along: npt.NDArray[np.float64],
    boundaries: npt.NDArray[np.float64] | None,
    num_slices: int,
) -> npt.NDArray[np.float64]:
    if boundaries is not None:
        resolved = np.asarray(boundaries, dtype=np.float64)
        if resolved.ndim != 1 or resolved.size < 2:
            raise ValueError("Slice boundaries must be a 1D array with at least two entries.")
        if not np.all(np.diff(resolved) > 0.0):
            raise ValueError("Slice boundaries must be strictly increasing.")
    else:
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
