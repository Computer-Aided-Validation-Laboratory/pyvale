from __future__ import annotations

from copy import copy
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

from pyvale.vfm.constparam import ConstitutiveParameter
from pyvale.vfm.dof import DegreeOfFreedom
from pyvale.vfm.experimentdata import ExperimentData, SpecimenGeometry
from pyvale.vfm import slicewise_utils
from pyvale.vfm.spatialparam import ISpatialParameterisation


SliceAreaPartition = slicewise_utils.SliceAreaPartition
SliceAssignmentPartition = slicewise_utils.SliceAssignmentPartition
SliceAxis = slicewise_utils.SliceAxis
SliceConfig = slicewise_utils.SliceConfig
SlicePartition = slicewise_utils.SlicePartition
build_slice_area_partition = slicewise_utils.build_slice_area_partition
build_slice_partition = slicewise_utils.build_slice_partition
plot_slice_area_partition_diagnostic = slicewise_utils.plot_slice_area_partition_diagnostic
resolve_slice_partition = slicewise_utils.resolve_slice_partition


@dataclass(slots=True)
class SupportSlice:
    """Resolved slice support that may be shared across parameters and metrics."""

    slice_partition: SliceAreaPartition | None = None
    slice_config: SliceConfig | None = None

    def __post_init__(self) -> None:
        if self.slice_partition is None and self.slice_config is None:
            raise ValueError("Provide either slice_partition or slice_config.")

    def prepare_from_specimen_geometry(
        self,
        specimen_geometry: SpecimenGeometry,
    ) -> None:
        self.slice_partition = resolve_slice_partition(
            specimen_geometry,
            slice_partition=self.slice_partition,
            slice_config=self.slice_config,
        )

    def prepare(
        self,
        experiment_data: ExperimentData,
    ) -> None:
        self.prepare_from_specimen_geometry(experiment_data.specimen_geometry)

    def get_num_degrees_of_freedom(self) -> int:
        return 0

    def collect_degrees_of_freedom(self) -> list[DegreeOfFreedom]:
        return []

    def update_from_degrees_of_freedom(
        self,
        degrees_of_freedom: list[DegreeOfFreedom] | npt.NDArray[np.float64],
    ) -> None:
        return


@dataclass(slots=True)
class SliceWiseSpatialParameterisation(ISpatialParameterisation):
    """Piecewise-constant parameterisation with one value per slice."""

    support: SupportSlice | None = None
    slice_partition: SliceAssignmentPartition | None = None
    slice_config: SliceConfig | None = None
    values: list[float | DegreeOfFreedom | None] | None = None

    def __post_init__(self) -> None:
        if self.support is None:
            self.support = SupportSlice(
                slice_partition=self.slice_partition,
                slice_config=self.slice_config,
            )
        elif self.slice_partition is not None or self.slice_config is not None:
            raise ValueError(
                "Provide either support or slice_partition/slice_config."
            )

        self._sync_from_support()
        if self.slice_partition is not None:
            self._ensure_values_match_partition()

    def _sync_from_support(self) -> None:
        if self.support is None:
            return
        self.slice_partition = self.support.slice_partition
        self.slice_config = self.support.slice_config

    def set_support(
        self,
        support: SupportSlice,
    ) -> None:
        self.support = support
        self._sync_from_support()
        self._ensure_values_match_partition()

    def initialise_slice_partition(
        self,
        specimen_geometry: SpecimenGeometry,
    ) -> None:
        """Resolve the slice partition once specimen geometry is available."""

        assert self.support is not None
        self.support.prepare_from_specimen_geometry(specimen_geometry)
        self._sync_from_support()
        self._ensure_values_match_partition()

    def _ensure_values_match_partition(self) -> None:
        if self.slice_partition is None:
            return

        if self.values is None:
            self.values = [None] * self.slice_partition.num_slices

        if len(self.values) != self.slice_partition.num_slices:
            raise ValueError(
                f"Expected {self.slice_partition.num_slices} slice values, got {len(self.values)}."
            )

    def get_num_degrees_of_freedom(self) -> int:
        if self.values is None:
            if self.slice_partition is None:
                return 0
            return self.slice_partition.num_slices
        return sum(isinstance(value, DegreeOfFreedom) or value is None for value in self.values)

    def initialise_from_constitutive_parameter(
        self,
        constitutive_parameter: ConstitutiveParameter,
    ) -> None:
        if self.slice_partition is None:
            raise RuntimeError(
                "SliceWiseSpatialParameterisation slice partition has not been resolved. "
                "Call initialise_slice_partition(...) before initialise_from_constitutive_parameter(...)."
            )

        self._ensure_values_match_partition()
        assert self.values is not None

        parameter_values = np.asarray(constitutive_parameter.map, dtype=np.float64)
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

    def update_from_constitutive_parameter(
        self,
        constitutive_parameter: ConstitutiveParameter,
    ) -> None:
        self.initialise_from_constitutive_parameter(constitutive_parameter)

    def to_map(
        self,
        size: npt.NDArray[np.uint32],
    ) -> npt.NDArray[np.float64]:
        if self.slice_partition is None:
            raise RuntimeError("Slice partition has not been resolved.")

        map_shape = (int(size[0]), int(size[1]))
        if map_shape != self.slice_partition.slice_id_map.shape:
            raise ValueError(
                f"Requested map shape {map_shape} does not match the slice partition shape "
                f"{self.slice_partition.slice_id_map.shape}."
            )

        if self.values is None:
            raise RuntimeError("values have not been initialised.")

        slice_values = np.asarray([_resolve_slice_value(value) for value in self.values], dtype=np.float64)
        parameter_map = np.full(map_shape, float(np.mean(slice_values)), dtype=np.float64)
        for slice_index, slice_value in enumerate(slice_values):
            parameter_map[self.slice_partition.slice_id_map == slice_index] = slice_value
        return parameter_map

    def collect_degrees_of_freedom(self) -> list[DegreeOfFreedom]:
        if self.values is None:
            raise RuntimeError("values have not been initialised.")

        dofs: list[DegreeOfFreedom] = []
        for value in self.values:
            if isinstance(value, DegreeOfFreedom):
                dofs.append(copy(value))
            elif value is None:
                raise ValueError(
                    "SliceWiseSpatialParameterisation must be initialised before collecting DOFs."
                )
        return dofs

    def update_from_degrees_of_freedom(
        self,
        degrees_of_freedom: list[DegreeOfFreedom] | npt.NDArray[np.float64],
    ) -> None:
        if self.values is None:
            raise RuntimeError("values have not been initialised.")

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

    def should_perform_refinement(self) -> bool:
        return False


def _resolve_slice_value(value: float | DegreeOfFreedom | None) -> float:
    if isinstance(value, DegreeOfFreedom):
        return float(value.value)
    if value is None:
        raise ValueError(
            "SliceWiseSpatialParameterisation must be initialised before converting to a map."
        )
    return float(value)


SlicewiseSpatialParameterisation = SliceWiseSpatialParameterisation
