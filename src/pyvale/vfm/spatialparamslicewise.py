from __future__ import annotations

from copy import copy
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

from pyvale.vfm.constparam import ConstitutiveParameter
from pyvale.vfm.dof import DegreeOfFreedom
from pyvale.vfm.slicepartition import SlicePartition
from pyvale.vfm.spatialparam import ISpatialParameterisation


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


SlicewiseSpatialParameterisation = SliceWiseSpatialParameterisation
