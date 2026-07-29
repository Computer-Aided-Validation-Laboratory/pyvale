from __future__ import annotations

from copy import copy
from dataclasses import dataclass, field

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
    """Resolved slice support that may be shared across parameters and metrics.

    `slice_partition` becomes available once experiment geometry is known.
    `slice_config` retains the declaration that can be used to rebuild the
    partition after refinement or when a fresh runtime is prepared.
    """

    slice_partition: SliceAreaPartition | None = None
    slice_config: SliceConfig | None = None
    refine: bool = False
    merge_parameter_tolerance: float = 0.05
    split_error_threshold: float = 0.1
    max_refinements: int = 1
    _num_refinements: int = field(
        default=0,
        init=False,
        repr=False,
    )

    def __post_init__(self) -> None:
        if self.slice_partition is None and self.slice_config is None:
            raise ValueError("Provide either slice_partition or slice_config.")
        if self.merge_parameter_tolerance < 0.0:
            raise ValueError("merge_parameter_tolerance must be non-negative.")
        if self.split_error_threshold < 0.0:
            raise ValueError("split_error_threshold must be non-negative.")
        if self.max_refinements < 0:
            raise ValueError("max_refinements must be non-negative.")

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

    def matches_config(
        self,
        slice_config: SliceConfig | None,
    ) -> bool:
        """Return True when this support is compatible with the given config."""

        if slice_config is None:
            return False

        if self.slice_partition is not None:
            return slicewise_utils.slice_partition_matches_config(
                self.slice_partition,
                slice_config,
            )

        support_config = self.slice_config
        if support_config is None or support_config.axis != slice_config.axis:
            return False

        if support_config.boundaries is not None or slice_config.boundaries is not None:
            if support_config.boundaries is None or slice_config.boundaries is None:
                return False
            return (
                support_config.boundaries.shape == slice_config.boundaries.shape
                and np.allclose(
                    support_config.boundaries,
                    slice_config.boundaries,
                )
            )

        return support_config.num_slices == slice_config.num_slices

    def is_equivalent_to(
        self,
        other: "SupportSlice",
    ) -> bool:
        """Return True when two support declarations describe the same slices."""

        if self is other:
            return True

        if self.slice_partition is not None and other.slice_partition is not None:
            return slicewise_utils.slice_partitions_are_equivalent(
                self.slice_partition,
                other.slice_partition,
            )

        if self.slice_partition is not None:
            return self.matches_config(other.slice_config)

        if other.slice_partition is not None:
            return other.matches_config(self.slice_config)

        return self.matches_config(other.slice_config)

    def get_num_degrees_of_freedom(self) -> int:
        return 0

    def collect_degrees_of_freedom(self) -> list[DegreeOfFreedom]:
        return []

    def update_from_degrees_of_freedom(
        self,
        degrees_of_freedom: list[DegreeOfFreedom] | npt.NDArray[np.float64],
    ) -> None:
        return

    def should_perform_refinement(
        self,
        *,
        parameter_maps: dict[str, npt.NDArray[np.float64]],
        spatial_parameterisations: dict[str, list[ISpatialParameterisation]],
        force_error_ratio: npt.NDArray[np.float64] | None = None,
    ) -> bool:
        """Return True when the fresh post-solve state changes the partition."""

        if not self.refine or self._num_refinements >= self.max_refinements:
            return False
        return self._build_refined_boundaries(
            parameter_maps=parameter_maps,
            spatial_parameterisations=spatial_parameterisations,
            force_error_ratio=force_error_ratio,
        ) is not None

    def perform_refinement(
        self,
        *,
        parameter_maps: dict[str, npt.NDArray[np.float64]],
        spatial_parameterisations: dict[str, list[ISpatialParameterisation]],
        force_error_ratio: npt.NDArray[np.float64] | None = None,
    ) -> None:
        """Update slice boundaries using current maps and force diagnostics."""

        refined_boundaries = self._build_refined_boundaries(
            parameter_maps=parameter_maps,
            spatial_parameterisations=spatial_parameterisations,
            force_error_ratio=force_error_ratio,
        )
        if refined_boundaries is None:
            return

        assert self.slice_partition is not None
        self.slice_config = SliceConfig(
            axis=self.slice_partition.axis,
            boundaries=refined_boundaries,
        )
        self.slice_partition = None
        self._num_refinements += 1

    def _build_refined_boundaries(
        self,
        *,
        parameter_maps: dict[str, npt.NDArray[np.float64]],
        spatial_parameterisations: dict[str, list[ISpatialParameterisation]],
        force_error_ratio: npt.NDArray[np.float64] | None,
    ) -> npt.NDArray[np.float64] | None:
        if self.slice_partition is None:
            return None

        num_slices = self.slice_partition.num_slices
        merge_boundary_mask = self._get_merge_boundary_mask(
            parameter_maps=parameter_maps,
            spatial_parameterisations=spatial_parameterisations,
        )
        split_slice_mask = self._get_split_slice_mask(force_error_ratio)
        if merge_boundary_mask is None:
            merge_boundary_mask = np.zeros(max(num_slices - 1, 0), dtype=bool)
        if split_slice_mask is None:
            split_slice_mask = np.zeros(num_slices, dtype=bool)

        # A high-error slice asks for more local resolution, so keep its
        # existing neighbours even if the fitted parameter values are similar.
        if merge_boundary_mask.size > 0:
            merge_boundary_mask = (
                merge_boundary_mask
                & ~split_slice_mask[:-1]
                & ~split_slice_mask[1:]
            )

        if not np.any(merge_boundary_mask) and not np.any(split_slice_mask):
            return None

        old_boundaries = self.slice_partition.boundaries
        new_boundaries = [float(old_boundaries[0])]
        for slice_index in range(num_slices):
            if split_slice_mask[slice_index]:
                new_boundaries.append(
                    float(
                        0.5
                        * (
                            old_boundaries[slice_index]
                            + old_boundaries[slice_index + 1]
                        )
                    )
                )
            if (
                slice_index < num_slices - 1
                and not merge_boundary_mask[slice_index]
            ):
                new_boundaries.append(float(old_boundaries[slice_index + 1]))
        new_boundaries.append(float(old_boundaries[-1]))

        refined_boundaries = np.asarray(new_boundaries, dtype=np.float64)
        if (
            refined_boundaries.shape == old_boundaries.shape
            and np.allclose(refined_boundaries, old_boundaries)
        ):
            return None
        return refined_boundaries

    def _get_merge_boundary_mask(
        self,
        *,
        parameter_maps: dict[str, npt.NDArray[np.float64]],
        spatial_parameterisations: dict[str, list[ISpatialParameterisation]],
    ) -> npt.NDArray[np.bool_] | None:
        if self.slice_partition is None or self.slice_partition.num_slices < 2:
            return None

        parameter_names = self._get_refined_parameter_names(
            spatial_parameterisations
        )
        if not parameter_names:
            return None

        merge_boundary_mask = np.ones(
            self.slice_partition.num_slices - 1,
            dtype=bool,
        )
        for parameter_name in parameter_names:
            if parameter_name not in parameter_maps:
                raise ValueError(
                    f"No parameter map was supplied for '{parameter_name}'."
                )
            slice_values = self._calculate_slice_means(
                parameter_maps[parameter_name]
            )
            value_scale = np.maximum(
                np.maximum(
                    np.abs(slice_values[:-1]),
                    np.abs(slice_values[1:]),
                ),
                1.0e-12,
            )
            relative_difference = (
                np.abs(slice_values[:-1] - slice_values[1:]) / value_scale
            )
            merge_boundary_mask &= (
                np.isfinite(relative_difference)
                & (relative_difference <= self.merge_parameter_tolerance)
            )

        if not np.any(merge_boundary_mask):
            return None
        return merge_boundary_mask

    def _get_split_slice_mask(
        self,
        force_error_ratio: npt.NDArray[np.float64] | None,
    ) -> npt.NDArray[np.bool_] | None:
        if self.slice_partition is None or force_error_ratio is None:
            return None

        resolved_force_error_ratio = np.asarray(
            force_error_ratio,
            dtype=np.float64,
        )
        if resolved_force_error_ratio.shape != (self.slice_partition.num_slices,):
            raise ValueError(
                "Force reconstruction error ratio shape does not match the "
                f"slice partition: {resolved_force_error_ratio.shape} vs "
                f"({self.slice_partition.num_slices},)."
            )

        split_slice_mask = (
            np.isfinite(resolved_force_error_ratio)
            & (resolved_force_error_ratio > self.split_error_threshold)
        )
        if not np.any(split_slice_mask):
            return None
        return split_slice_mask

    def _get_refined_parameter_names(
        self,
        spatial_parameterisations: dict[str, list[ISpatialParameterisation]],
    ) -> tuple[str, ...]:
        parameter_names: list[str] = []
        for parameter_name, parameterisation_list in spatial_parameterisations.items():
            if any(
                getattr(parameterisation, "support", None) is self
                and parameterisation.get_num_degrees_of_freedom() > 0
                for parameterisation in parameterisation_list
            ):
                parameter_names.append(parameter_name)
        return tuple(parameter_names)

    def _calculate_slice_means(
        self,
        parameter_map: npt.NDArray[np.float64],
    ) -> npt.NDArray[np.float64]:
        if self.slice_partition is None:
            raise RuntimeError("Slice partition has not been resolved.")
        if parameter_map.shape != self.slice_partition.slice_id_map.shape:
            raise ValueError(
                "Parameter map shape does not match the slice partition shape: "
                f"{parameter_map.shape} vs {self.slice_partition.slice_id_map.shape}."
            )

        slice_means = np.full(
            self.slice_partition.num_slices,
            np.nan,
            dtype=np.float64,
        )
        for slice_index in range(self.slice_partition.num_slices):
            slice_mask = self.slice_partition.slice_id_map == slice_index
            finite_values = parameter_map[slice_mask & np.isfinite(parameter_map)]
            if finite_values.size > 0:
                slice_means[slice_index] = float(np.mean(finite_values))
        return slice_means


@dataclass(slots=True, init=False)
class SliceWiseSpatialParameterisation(ISpatialParameterisation):
    """Piecewise-constant parameterisation with one value per slice."""

    support: SupportSlice
    values: list[float | DegreeOfFreedom | None] | None = None

    def __init__(
        self,
        support: SupportSlice | None = None,
        slice_partition: SliceAssignmentPartition | None = None,
        slice_config: SliceConfig | None = None,
        values: list[float | DegreeOfFreedom | None] | None = None,
    ) -> None:
        if support is None:
            support = SupportSlice(
                slice_partition=slice_partition,
                slice_config=slice_config,
            )
        elif slice_partition is not None or slice_config is not None:
            raise ValueError(
                "Provide either support or slice_partition/slice_config."
            )

        self.support = support
        self.values = values
        if self.slice_partition is not None:
            self._ensure_values_match_partition()

    @property
    def slice_partition(self) -> SliceAssignmentPartition | None:
        return self.support.slice_partition

    @property
    def slice_config(self) -> SliceConfig | None:
        return self.support.slice_config

    def set_support(
        self,
        support: SupportSlice,
    ) -> None:
        self.support = support
        self._ensure_values_match_partition()

    def initialise_slice_partition(
        self,
        specimen_geometry: SpecimenGeometry,
    ) -> None:
        """Resolve the slice partition once specimen geometry is available."""

        assert self.support is not None
        self.support.prepare_from_specimen_geometry(specimen_geometry)
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

        self._resize_values_for_initialisation()
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

    def _resize_values_for_initialisation(self) -> None:
        """Resize slice values before fitting onto a refined support."""

        if self.slice_partition is None or self.values is None:
            return
        if len(self.values) == self.slice_partition.num_slices:
            return

        old_values = self.values
        all_values_fixed = all(
            value is not None and not isinstance(value, DegreeOfFreedom)
            for value in old_values
        )
        placeholder = 0.0 if all_values_fixed else None
        self.values = [placeholder] * self.slice_partition.num_slices

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
