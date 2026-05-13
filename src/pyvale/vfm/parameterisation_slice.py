from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import numpy.typing as npt

from pyvale.vfm.project_definition import (
    ParameterDefinition,
    ParameterisationSpec,
    PhaseResult,
    TestData,
    resolve_parameter_initial_value,
)
from pyvale.vfm.spatial_parameterisation import BaseParameterisation, ParameterisationDof


@dataclass(slots=True)
class SliceSubdomain:
    """One contiguous slice extracted from the full test-data grid."""

    slice_index: int
    row_slice: slice
    col_slice: slice
    local_test_data: TestData


@dataclass(slots=True)
class SlicePartition:
    """Shared slicewise layout used by parameterisations and metrics."""

    constant_coordinate: str
    varying_coordinate: str
    num_slices: int
    slice_index_map: npt.NDArray[np.int64]
    slice_masks: tuple[npt.NDArray[np.bool_], ...]
    slice_widths: npt.NDArray[np.float64]
    slice_subdomains: tuple[SliceSubdomain, ...]


def normalise_constant_coordinate(options: dict[str, Any]) -> str:
    """Resolve the preferred slicewise layout option to `x` or `y`."""

    constant_coordinate = options.get("constant_coordinate")
    if constant_coordinate is not None:
        axis = str(constant_coordinate).strip().lower()
        if axis in {"x", "y"}:
            return axis
        raise ValueError(
            "Slicewise option 'constant_coordinate' must be 'x' or 'y'."
        )

    direction = options.get("direction", "x")
    direction = str(direction).strip().lower()
    if direction == "x":
        return "y"
    if direction == "y":
        return "x"

    raise ValueError("Slicewise option 'direction' must be 'x' or 'y'.")


def build_slice_partition(
    test_data: TestData,
    options: dict[str, Any],
) -> SlicePartition:
    """Construct contiguous row/column slice groups from the measured grid."""

    constant_coordinate = normalise_constant_coordinate(options)
    varying_coordinate = "y" if constant_coordinate == "x" else "x"
    num_slices = int(options.get("num_slices", 1))
    if num_slices < 1:
        raise ValueError("Slicewise option 'num_slices' must be at least 1.")

    specimen_mask = np.asarray(test_data.specimen_mask, dtype=bool)
    if specimen_mask.ndim != 2:
        raise ValueError("test_data.specimen_mask must be a 2D array.")

    if constant_coordinate == "x":
        active_indices = np.where(np.any(specimen_mask, axis=1))[0]
        coordinate_line = np.asarray(np.nanmean(test_data.y, axis=1), dtype=np.float64)
        line_edges = _compute_centroid_edges(coordinate_line)
    else:
        active_indices = np.where(np.any(specimen_mask, axis=0))[0]
        coordinate_line = np.asarray(np.nanmean(test_data.x, axis=0), dtype=np.float64)
        line_edges = _compute_centroid_edges(coordinate_line)

    if active_indices.size == 0:
        raise ValueError("Cannot build slicewise layout for an empty specimen mask.")
    if num_slices > active_indices.size:
        raise ValueError(
            f"Requested {num_slices} slices but only {active_indices.size} "
            f"{varying_coordinate}-lines contain specimen points."
        )

    index_groups = tuple(
        np.asarray(group, dtype=np.int64)
        for group in np.array_split(active_indices, num_slices)
    )
    if any(group.size == 0 for group in index_groups):
        raise ValueError("At least one generated slice is empty.")

    slice_index_map = np.full(specimen_mask.shape, -1, dtype=np.int64)
    slice_masks: list[npt.NDArray[np.bool_]] = []
    slice_widths: list[float] = []
    slice_subdomains: list[SliceSubdomain] = []

    for slice_index, group in enumerate(index_groups):
        slice_mask = np.zeros_like(specimen_mask, dtype=bool)
        if constant_coordinate == "x":
            slice_mask[group, :] = specimen_mask[group, :]
            width = float(line_edges[group[-1] + 1] - line_edges[group[0]])
        else:
            slice_mask[:, group] = specimen_mask[:, group]
            width = float(line_edges[group[-1] + 1] - line_edges[group[0]])

        if not np.any(slice_mask):
            raise ValueError(f"Generated slice {slice_index} contains no specimen points.")

        slice_index_map[slice_mask] = slice_index
        slice_masks.append(slice_mask)
        slice_widths.append(width)
        slice_subdomains.append(
            _build_slice_subdomain(
                test_data=test_data,
                slice_mask=slice_mask,
                slice_index=slice_index,
            )
        )

    return SlicePartition(
        constant_coordinate=constant_coordinate,
        varying_coordinate=varying_coordinate,
        num_slices=num_slices,
        slice_index_map=slice_index_map,
        slice_masks=tuple(slice_masks),
        slice_widths=np.asarray(slice_widths, dtype=np.float64),
        slice_subdomains=tuple(slice_subdomains),
    )


def slice_partitions_match(left: SlicePartition, right: SlicePartition) -> bool:
    return (
        left.constant_coordinate == right.constant_coordinate
        and left.varying_coordinate == right.varying_coordinate
        and left.num_slices == right.num_slices
        and np.array_equal(left.slice_index_map, right.slice_index_map)
        and np.allclose(left.slice_widths, right.slice_widths)
    )


@dataclass(slots=True)
class SliceWiseParameterisation(BaseParameterisation):
    parameter_name: str
    options: dict[str, Any] = field(default_factory=dict)
    initial_value: float | npt.NDArray[np.float64] | None = None
    lower_bound: float = 0.0
    upper_bound: float = 0.0
    initialise_from: str = "initial_value"
    dofs: list[ParameterisationDof] = field(default_factory=list)
    partition: SlicePartition | None = None
    kind: str = "slicewise"

    def prepare(self, test_data: TestData) -> None:
        if self.partition is None:
            self.partition = build_slice_partition(test_data, self.options)

        if isinstance(self.initial_value, np.ndarray):
            for dof, slice_mask in zip(
                self.dofs,
                self.partition.slice_masks,
                strict=True,
            ):
                finite_values = self.initial_value[slice_mask]
                if finite_values.size == 0:
                    continue
                dof.value = float(np.nanmean(finite_values))

    def initialise(
        self,
        test_data: TestData,
        source_map: npt.NDArray[np.float64] | None = None,
    ) -> None:
        if source_map is None or self.initialise_from == "initial_value":
            return

        if self.partition is None:
            self.partition = build_slice_partition(test_data, self.options)

        for dof, slice_mask in zip(
            self.dofs,
            self.partition.slice_masks,
            strict=True,
        ):
            finite_values = source_map[slice_mask]
            if finite_values.size == 0:
                continue
            dof.value = float(np.nanmean(finite_values))

    def collect_dofs(self) -> list[ParameterisationDof]:
        return self.dofs

    def slice_dof(self, slice_index: int) -> ParameterisationDof:
        return self.dofs[slice_index]

    def to_map(self, test_data: TestData) -> npt.NDArray[np.float64]:
        if self.partition is None:
            self.partition = build_slice_partition(test_data, self.options)

        parameter_map = np.full(test_data.specimen_mask.shape, np.nan, dtype=np.float64)
        for dof, slice_mask in zip(
            self.dofs,
            self.partition.slice_masks,
            strict=True,
        ):
            parameter_map[slice_mask] = dof.value

        parameter_map[~test_data.specimen_mask] = np.nan
        return parameter_map


def build_slice_parameterisation(
    parameter_name: str,
    parameter_definition: ParameterDefinition,
    parameterisation_spec: ParameterisationSpec,
    previous_result: PhaseResult | None = None,
) -> BaseParameterisation:
    options = dict(parameterisation_spec.options)
    num_slices = int(options.get("num_slices", 1))
    if num_slices < 1:
        raise ValueError(
            f"Parameter '{parameter_name}' needs at least one slice."
        )

    initial_value = options.get("value")
    if initial_value is None:
        initial_value = resolve_parameter_initial_value(parameter_definition)
    if initial_value is None:
        raise ValueError(
            f"Parameter '{parameter_name}' needs an initial value for a "
            "slicewise parameterisation."
        )

    lower_bound = options.get("lower_bound", parameter_definition.lower_bound)
    upper_bound = options.get("upper_bound", parameter_definition.upper_bound)
    if lower_bound is None or upper_bound is None:
        raise ValueError(
            f"Parameter '{parameter_name}' needs lower and upper bounds for "
            "a slicewise parameterisation."
        )

    initial_scalar = (
        float(np.nanmean(initial_value))
        if isinstance(initial_value, np.ndarray)
        else float(initial_value)
    )
    dofs = [
        ParameterisationDof(
            uid=f"{parameter_name}.slicewise.slice_{slice_index}",
            group=f"slice_{slice_index}",
            value=initial_scalar,
            lower_bound=float(lower_bound),
            upper_bound=float(upper_bound),
            active=True,
        )
        for slice_index in range(num_slices)
    ]

    return SliceWiseParameterisation(
        parameter_name=parameter_name,
        options=options,
        initial_value=initial_value,
        lower_bound=float(lower_bound),
        upper_bound=float(upper_bound),
        initialise_from=str(options.get("initialise_from", "initial_value")),
        dofs=dofs,
    )


def _compute_centroid_edges(
    centroid_coordinates: npt.NDArray[np.float64],
) -> npt.NDArray[np.float64]:
    if centroid_coordinates.ndim != 1:
        raise ValueError("Expected a 1D centroid coordinate line.")
    if centroid_coordinates.size == 0:
        raise ValueError("Need at least one centroid coordinate.")

    edges = np.empty(centroid_coordinates.size + 1, dtype=np.float64)
    if centroid_coordinates.size == 1:
        edges[0] = centroid_coordinates[0] - 0.5
        edges[1] = centroid_coordinates[0] + 0.5
        return edges

    edges[1:-1] = 0.5 * (
        centroid_coordinates[:-1] + centroid_coordinates[1:]
    )
    edges[0] = centroid_coordinates[0] - (edges[1] - centroid_coordinates[0])
    edges[-1] = centroid_coordinates[-1] + (
        centroid_coordinates[-1] - edges[-2]
    )
    return edges


def _build_slice_subdomain(
    test_data: TestData,
    slice_mask: npt.NDArray[np.bool_],
    slice_index: int,
) -> SliceSubdomain:
    row_indices = np.where(np.any(slice_mask, axis=1))[0]
    col_indices = np.where(np.any(slice_mask, axis=0))[0]
    if row_indices.size == 0 or col_indices.size == 0:
        raise ValueError(f"Slice {slice_index} contains no points.")

    row_slice = slice(int(row_indices[0]), int(row_indices[-1]) + 1)
    col_slice = slice(int(col_indices[0]), int(col_indices[-1]) + 1)
    local_mask = slice_mask[row_slice, col_slice]

    local_test_data = TestData(
        x=test_data.x[row_slice, col_slice].copy(),
        y=test_data.y[row_slice, col_slice].copy(),
        specimen_mask=local_mask.copy(),
        area=test_data.area[row_slice, col_slice].copy(),
        strain=test_data.strain[:, :, row_slice, col_slice].copy(),
        force=test_data.force.copy(),
        time=test_data.time.copy(),
        thickness=test_data.thickness,
        boundary_conditions=test_data.boundary_conditions,
        source_path=test_data.source_path,
    )

    return SliceSubdomain(
        slice_index=slice_index,
        row_slice=row_slice,
        col_slice=col_slice,
        local_test_data=local_test_data,
    )
