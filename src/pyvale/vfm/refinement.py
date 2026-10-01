from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np
import numpy.typing as npt

from pyvale.vfm.constlaw import IConstitutiveLaw
from pyvale.vfm.constparam import ConstitutiveParameter
from pyvale.vfm.experimentdata import ExperimentData
from pyvale.vfm.metricsliceforce import SliceWiseForceReconstructionMetric
from pyvale.vfm.spatialparam import ISpatialParameterisation
from pyvale.vfm.spatialparamslicewise import SliceConfig, SupportSlice

if TYPE_CHECKING:
    from pyvale.vfm.identification import PhaseRuntime


RefinementTarget = object | str | tuple[str, int]


@dataclass(slots=True)
class RefinementContext:
    """Solved-state data available to phase-level refinement policies."""

    experiment_data: ExperimentData
    constitutive_law: IConstitutiveLaw
    constitutive_parameters: dict[str, ConstitutiveParameter]
    parameter_map_size: npt.NDArray[np.uint32]
    parameter_maps: dict[str, npt.NDArray[np.float64]]
    metrics: list[object] = field(default_factory=list)
    objective_function: object | None = None
    objective_value: float | None = None
    _stress: npt.NDArray[np.float64] | None = field(
        default=None,
        init=False,
        repr=False,
    )

    @property
    def stress(self) -> npt.NDArray[np.float64]:
        if self._stress is None:
            self._stress = self.constitutive_law.calculate_stress(
                self.experiment_data.strain,
                self.parameter_maps,
            )
        return self._stress


class IRefinementAction(ABC):
    """A single structural change proposed after a phase solve."""

    @abstractmethod
    def apply(
        self,
        runtime: PhaseRuntime,
        context: RefinementContext,
    ) -> None:
        ...

    @property
    def terminal(self) -> bool:
        """Whether applying this action ends the current phase."""

        return False

    @property
    def accepts_current_solve(self) -> bool:
        """Whether the solved candidate remains the active phase model."""

        return True


class IRefinementPolicy(ABC):
    """Phase-level strategy that proposes one structural change at a time."""

    @abstractmethod
    def propose(
        self,
        runtime: PhaseRuntime,
        context: RefinementContext,
    ) -> IRefinementAction | None:
        ...

@dataclass(slots=True)
class SliceMergeSplitRefinement(IRefinementPolicy):
    """Merge similar neighbouring slices and split high-error slices."""

    target: RefinementTarget
    max_refinements: int = 1
    merge_parameter_tolerance: float = 0.05
    split_error_threshold: float = 0.1
    _num_refinements: int = field(
        default=0,
        init=False,
        repr=False,
    )

    def __post_init__(self) -> None:
        if self.max_refinements < 0:
            raise ValueError("max_refinements must be non-negative.")
        if self.merge_parameter_tolerance < 0.0:
            raise ValueError("merge_parameter_tolerance must be non-negative.")
        if self.split_error_threshold < 0.0:
            raise ValueError("split_error_threshold must be non-negative.")

    def propose(
        self,
        runtime: PhaseRuntime,
        context: RefinementContext,
    ) -> IRefinementAction | None:
        if self._num_refinements >= self.max_refinements:
            return None

        support = runtime.resolve_support_target(self.target)
        if not isinstance(support, SupportSlice):
            raise TypeError(
                "SliceMergeSplitRefinement target must resolve to SupportSlice."
            )

        force_error_ratio = _evaluate_slice_force_error_ratio(
            support,
            runtime,
            context,
        )
        refined_boundaries = _build_refined_slice_boundaries(
            support=support,
            parameter_maps=context.parameter_maps,
            spatial_parameterisations=runtime.spatial_parameterisations,
            force_error_ratio=force_error_ratio,
            merge_parameter_tolerance=self.merge_parameter_tolerance,
            split_error_threshold=self.split_error_threshold,
        )
        if refined_boundaries is None:
            return None

        return SliceMergeSplitAction(
            support=support,
            refined_boundaries=refined_boundaries,
            policy=self,
        )

    def _record_refinement(self) -> None:
        self._num_refinements += 1


@dataclass(slots=True)
class SliceMergeSplitAction(IRefinementAction):
    support: SupportSlice
    refined_boundaries: npt.NDArray[np.float64]
    policy: SliceMergeSplitRefinement

    def apply(
        self,
        runtime: PhaseRuntime,
        context: RefinementContext,
    ) -> None:
        if self.support.slice_partition is None:
            return

        self.support.slice_config = SliceConfig(
            axis=self.support.slice_partition.axis,
            boundaries=self.refined_boundaries,
        )
        self.support.slice_partition = None
        self.policy._record_refinement()


def _evaluate_slice_force_error_ratio(
    support: SupportSlice,
    runtime: PhaseRuntime,
    context: RefinementContext,
) -> npt.NDArray[np.float64] | None:
    for metric in runtime.metrics:
        if not isinstance(metric, SliceWiseForceReconstructionMetric):
            continue
        if metric.support is not support:
            continue
        result = metric.evaluate_force_recon_error(
            context.stress,
            context.experiment_data,
        )
        return result.weighted_temporal_rms
    return None


def _build_refined_slice_boundaries(
    *,
    support: SupportSlice,
    parameter_maps: dict[str, npt.NDArray[np.float64]],
    spatial_parameterisations: dict[str, list[ISpatialParameterisation]],
    force_error_ratio: npt.NDArray[np.float64] | None,
    merge_parameter_tolerance: float,
    split_error_threshold: float,
) -> npt.NDArray[np.float64] | None:
    if support.slice_partition is None:
        return None

    num_slices = support.slice_partition.num_slices
    merge_boundary_mask = _get_merge_boundary_mask(
        support=support,
        parameter_maps=parameter_maps,
        spatial_parameterisations=spatial_parameterisations,
        merge_parameter_tolerance=merge_parameter_tolerance,
    )
    split_slice_mask = _get_split_slice_mask(
        support,
        force_error_ratio,
        split_error_threshold,
    )
    if merge_boundary_mask is None:
        merge_boundary_mask = np.zeros(max(num_slices - 1, 0), dtype=bool)
    if split_slice_mask is None:
        split_slice_mask = np.zeros(num_slices, dtype=bool)

    if merge_boundary_mask.size > 0:
        merge_boundary_mask = (
            merge_boundary_mask
            & ~split_slice_mask[:-1]
            & ~split_slice_mask[1:]
        )

    if not np.any(merge_boundary_mask) and not np.any(split_slice_mask):
        return None

    old_boundaries = support.slice_partition.boundaries
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
    *,
    support: SupportSlice,
    parameter_maps: dict[str, npt.NDArray[np.float64]],
    spatial_parameterisations: dict[str, list[ISpatialParameterisation]],
    merge_parameter_tolerance: float,
) -> npt.NDArray[np.bool_] | None:
    if support.slice_partition is None or support.slice_partition.num_slices < 2:
        return None

    parameter_names = _get_refined_parameter_names(
        support,
        spatial_parameterisations,
    )
    if not parameter_names:
        return None

    merge_boundary_mask = np.ones(
        support.slice_partition.num_slices - 1,
        dtype=bool,
    )
    for parameter_name in parameter_names:
        if parameter_name not in parameter_maps:
            raise ValueError(
                f"No parameter map was supplied for '{parameter_name}'."
            )
        slice_values = _calculate_slice_means(
            support,
            parameter_maps[parameter_name],
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
            & (relative_difference <= merge_parameter_tolerance)
        )

    if not np.any(merge_boundary_mask):
        return None
    return merge_boundary_mask


def _get_split_slice_mask(
    support: SupportSlice,
    force_error_ratio: npt.NDArray[np.float64] | None,
    split_error_threshold: float,
) -> npt.NDArray[np.bool_] | None:
    if support.slice_partition is None or force_error_ratio is None:
        return None

    resolved_force_error_ratio = np.asarray(
        force_error_ratio,
        dtype=np.float64,
    )
    if resolved_force_error_ratio.shape != (support.slice_partition.num_slices,):
        raise ValueError(
            "Force reconstruction error ratio shape does not match the "
            f"slice partition: {resolved_force_error_ratio.shape} vs "
            f"({support.slice_partition.num_slices},)."
        )

    split_slice_mask = (
        np.isfinite(resolved_force_error_ratio)
        & (resolved_force_error_ratio > split_error_threshold)
    )
    if not np.any(split_slice_mask):
        return None
    return split_slice_mask


def _get_refined_parameter_names(
    support: SupportSlice,
    spatial_parameterisations: dict[str, list[ISpatialParameterisation]],
) -> tuple[str, ...]:
    parameter_names: list[str] = []
    for parameter_name, parameterisation_list in spatial_parameterisations.items():
        if any(
            getattr(parameterisation, "support", None) is support
            and parameterisation.get_num_degrees_of_freedom() > 0
            for parameterisation in parameterisation_list
        ):
            parameter_names.append(parameter_name)
    return tuple(parameter_names)


def _calculate_slice_means(
    support: SupportSlice,
    parameter_map: npt.NDArray[np.float64],
) -> npt.NDArray[np.float64]:
    if support.slice_partition is None:
        raise RuntimeError("Slice partition has not been resolved.")
    if parameter_map.shape != support.slice_partition.slice_id_map.shape:
        raise ValueError(
            "Parameter map shape does not match the slice partition shape: "
            f"{parameter_map.shape} vs {support.slice_partition.slice_id_map.shape}."
        )

    slice_means = np.full(
        support.slice_partition.num_slices,
        np.nan,
        dtype=np.float64,
    )
    for slice_index in range(support.slice_partition.num_slices):
        slice_mask = support.slice_partition.slice_id_map == slice_index
        finite_values = parameter_map[slice_mask & np.isfinite(parameter_map)]
        if finite_values.size > 0:
            slice_means[slice_index] = float(np.mean(finite_values))
    return slice_means
