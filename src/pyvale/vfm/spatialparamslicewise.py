from __future__ import annotations

from copy import copy
from dataclasses import dataclass
from typing import Literal

import numpy as np
import numpy.typing as npt

from pyvale.vfm.constparam import ConstitutiveParameter
from pyvale.vfm.dof import DegreeOfFreedom
from pyvale.vfm.experimentdata import EdgeConditions, EEdgeCondition, SpecimenGeometry
from pyvale.vfm.spatialparam import ISpatialParameterisation


SliceAxis = Literal["x", "y"] # only slices along x and y axes are support currently


@dataclass(slots=True, frozen=True)
class SliceConfig:
    """
    Configuration for slice-wise parameterisation.
    
    Axis can be either x or y and defines the direction along which the slices are made.
    Provide either `num_slices` or `boundaries`.
    - `num_slices`: number of evenly spaced slices along the valid ROI extent.
    - `boundaries`: explicit 1D array of slice boundary coordinates.
    If both are provided, num_slices must match the number of boundaries, and `boundaries` takes precedence.
    """
    axis: SliceAxis | None = None
    num_slices: int | None = None
    boundaries: npt.NDArray[np.float64] | None = None

    def __post_init__(self) -> None:
        has_num_slices = self.num_slices is not None
        has_boundaries = self.boundaries is not None
        if has_num_slices == has_boundaries and self.num_slices != len(self.boundaries) - 1:
            raise ValueError("Number of slices does not match the number of boundaries.Provide either num_slices or boundaries.")
        if has_num_slices and self.num_slices < 1:
            raise ValueError("num_slices must be at least 1.")


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




def build_slice_partition(
    specimen_geometry: SpecimenGeometry,
    edge_conditions: EdgeConditions | None = None,
    slice_config: SliceConfig | None = None,
) -> SlicePartition:
    config = SliceConfig() if slice_config is None else slice_config
    axis = config.axis
    
    if not np.any(specimen_geometry.specimen_mask):
        raise ValueError("No valid specimen points were available to build the slice partition.")

    coordinate_along = specimen_geometry.x if axis == "x" else specimen_geometry.y
    valid_coordinates_along = coordinate_along[specimen_geometry.specimen_mask]

    # Resolve boundaries as an array of slice boundary coordinates
    boundaries = _resolve_slice_boundaries(
        valid_coordinates_along,
        config.boundaries,
        config.num_slices,
    )

    # Compute span and centres of each slice (in physical coordinates)
    spans = np.diff(boundaries)
    centres = 0.5 * (boundaries[:-1] + boundaries[1:])

    # Determine which slice each valid point belongs to, and create a map of slice indices
    # Note: any points on boundaries are assigned to the slice on the higher index ("right")
    slice_indices = np.searchsorted(boundaries, valid_coordinates_along, side="right") - 1
    slice_indices = np.clip(slice_indices, 0, spans.size - 1)

    # Validate that the slice boundaries fully cover the valid ROI extent
    if np.any(valid_coordinates_along < boundaries[0]) or np.any(valid_coordinates_along > boundaries[-1]):
        raise ValueError("Slice boundaries do not fully cover the valid ROI extent.")

    # Create a map of slice indices for all points in the specimen geometry, with -1 for invalid points
    slice_id_map = np.full(specimen_geometry.x.shape, -1, dtype=np.int32) # init array of -1
    valid_flat_indices = np.nonzero(np.ravel(specimen_geometry.specimen_mask))[0] 
    slice_id_map.ravel()[valid_flat_indices] = slice_indices.astype(np.int32) # assign slice indices to valid points


    widths = np.zeros(spans.size, dtype=np.float64)
    areas = np.zeros(spans.size, dtype=np.float64)
    point_counts = np.zeros(spans.size, dtype=np.int64)
    slice_point_indices: list[npt.NDArray[np.int64]] = []
    pixel_area_flat = specimen_geometry.pixel_area.ravel()

    # For each slice, collect the flat indices of valid points that belong to that slice, and compute the area and width
    for slice_index in range(spans.size):
        flat_indices = valid_flat_indices[slice_indices == slice_index] # indices of valid points in this slice
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
        valid_point_mask=specimen_geometry.specimen_mask,
        slice_point_indices=tuple(slice_point_indices),
    )


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
    """
    Resolve slice boundaries based on provided boundaries or number of slices.
    If boundaries are provided, they are validated. If not, they are generated based on num_slices.
    """
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


SlicewiseSpatialParameterisation = SliceWiseSpatialParameterisation






# Optional future upgrade: automatic axis inference 
# if config.axis is None:
#     if edge_conditions is None:
#         raise ValueError("slice axis was not provided and could not be inferred without edge_conditions.")
#     axis = infer_loading_axis(edge_conditions)
# else:
#     axis = config.axis
#
# def infer_loading_axis(edge_conditions: EdgeConditions) -> SliceAxis:
#     x_has_traction = (
#         edge_conditions.min_x_edge.x is EEdgeCondition.Traction
#         or edge_conditions.max_x_edge.x is EEdgeCondition.Traction
#     )
#     y_has_traction = (
#         edge_conditions.min_y_edge.y is EEdgeCondition.Traction
#         or edge_conditions.max_y_edge.y is EEdgeCondition.Traction
#     )

#     if x_has_traction and y_has_traction:
#         raise ValueError("Could not infer a unique loading axis because both x and y edges have traction DOFs.")
#     if x_has_traction:
#         return "x"
#     if y_has_traction:
#         return "y"
#     raise ValueError("Could not infer the loading axis from edge conditions. Please set it explicitly.")
