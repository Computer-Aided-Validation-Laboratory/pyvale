from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sequence
import copy
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Literal

import numpy as np
import numpy.typing as npt
from scipy.ndimage import label, uniform_filter

from pyvale.vfm.constlaw import IConstitutiveLaw
from pyvale.vfm.constparam import ConstitutiveParameter
from pyvale.vfm.experimentdata import ExperimentData
from pyvale.vfm.metricsliceforce import SliceWiseForceReconstructionMetric
from pyvale.vfm.metricsbvf import calculate_local_parameter_stress_sensitivity
from pyvale.vfm.equilibriumgapaggregation import (
    EquilibriumGapAggregationResult,
    aggregate_equilibrium_gap_results,
    calculate_nan_rms,
    extract_equilibrium_gap_temporal_rms,
)
from pyvale.vfm.metricequilibriumgap import EquilibriumGapMetric
from pyvale.vfm.objectivefunccombinedfreegi import (
    CombinedForceAndEquilibriumGapObjective,
)
from pyvale.vfm.optimiserpatternsearch import OptimiserPatternSearch
from pyvale.vfm.spatialparam import ISpatialParameterisation, PhaseSpatialState
from pyvale.vfm.spatialparambasisfuncs import (
    BasisFunctionKernel,
    SpatialParameterisationBasisFunction,
    SupportBasis,
)
from pyvale.vfm.dof import DegreeOfFreedom
from pyvale.vfm.spatialparamknown import SpatialParameterisationKnown
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


@dataclass(slots=True)
class BasisAddRemoveRefinement(IRefinementPolicy):
    """Add or remove one basis function on a basis support per solve cycle."""

    target: RefinementTarget
    max_refinements: int = 1
    mode: Literal["add", "remove", "add_remove"] = "add"
    seed_parameter_name: str | None = None
    add_residual_threshold: float | None = None
    remove_height_threshold: float | None = None
    max_basis_functions: int | None = None
    min_basis_functions: int = 0
    _num_refinements: int = field(
        default=0,
        init=False,
        repr=False,
    )

    def __post_init__(self) -> None:
        if self.max_refinements < 0:
            raise ValueError("max_refinements must be non-negative.")
        if self.mode not in {"add", "remove", "add_remove"}:
            raise ValueError("mode must be 'add', 'remove', or 'add_remove'.")
        if self.min_basis_functions < 0:
            raise ValueError("min_basis_functions must be non-negative.")
        if self.max_basis_functions is not None and self.max_basis_functions < 0:
            raise ValueError("max_basis_functions must be non-negative.")

    def propose(
        self,
        runtime: PhaseRuntime,
        context: RefinementContext,
    ) -> IRefinementAction | None:
        if self._num_refinements >= self.max_refinements:
            return None

        support = runtime.resolve_support_target(self.target)
        if not isinstance(support, SupportBasis):
            raise TypeError(
                "BasisAddRemoveRefinement target must resolve to SupportBasis."
            )

        if self.mode in {"remove", "add_remove"}:
            removable_index = _find_removable_basis_index(
                runtime,
                support,
                self.remove_height_threshold,
                self.min_basis_functions,
            )
            if removable_index is not None:
                self._record_refinement()
                return RemoveBasisFunctionAction(
                    support=support,
                    kernel_index=removable_index,
                )

        if self.mode not in {"add", "add_remove"}:
            return None

        assert support.kernels is not None
        if (
            self.max_basis_functions is not None
            and len(support.kernels) >= self.max_basis_functions
        ):
            return None

        parameter_name, parameterisation = _resolve_basis_seed_parameterisation(
            runtime,
            support,
            self.target,
            self.seed_parameter_name,
        )
        target_map = context.parameter_maps[parameter_name]
        residual = target_map - parameterisation.to_map(context.parameter_map_size)
        if (
            self.add_residual_threshold is not None
            and np.nanmax(np.abs(residual)) <= self.add_residual_threshold
        ):
            return None

        parameter = context.constitutive_parameters[parameter_name]
        kernel, height = parameterisation._initialise_kernel(
            target_map,
            parameter.upper_bound - parameter.lower_bound,
        )
        self._record_refinement()
        return AddBasisFunctionAction(
            support=support,
            kernel=kernel,
            seed_parameterisation=parameterisation,
            seed_height=height,
        )

    def _record_refinement(self) -> None:
        self._num_refinements += 1


@dataclass(slots=True)
class AddBasisFunctionAction(IRefinementAction):
    support: SupportBasis
    kernel: object
    seed_parameterisation: SpatialParameterisationBasisFunction
    seed_height: DegreeOfFreedom
    screening_diagnostics: dict[str, object] | None = None

    def apply(
        self,
        runtime: PhaseRuntime,
        context: RefinementContext,
    ) -> None:
        assert self.support.kernels is not None
        self.support.kernels.append(self.kernel)

        for _, parameterisation in runtime.get_parameterisations_using_support(
            self.support
        ):
            if not isinstance(parameterisation, SpatialParameterisationBasisFunction):
                continue
            if parameterisation is self.seed_parameterisation:
                parameterisation.heights.append(self.seed_height)
            else:
                parameterisation.heights.append(None)


@dataclass(slots=True)
class ReplaceBasisFunctionAction(IRefinementAction):
    """Replace one seeded basis after optional multi-start screening."""

    support: SupportBasis
    kernel_index: int
    kernel: BasisFunctionKernel
    seed_parameterisation: SpatialParameterisationBasisFunction
    seed_height: DegreeOfFreedom
    screening_diagnostics: dict[str, object]

    def apply(
        self,
        runtime: PhaseRuntime,
        context: RefinementContext,
    ) -> None:
        _ = runtime, context
        assert self.support.kernels is not None
        self.support.kernels[self.kernel_index] = self.kernel
        self.seed_parameterisation.heights[self.kernel_index] = self.seed_height


@dataclass(slots=True)
class RemoveBasisFunctionAction(IRefinementAction):
    support: SupportBasis
    kernel_index: int

    def apply(
        self,
        runtime: PhaseRuntime,
        context: RefinementContext,
    ) -> None:
        assert self.support.kernels is not None
        del self.support.kernels[self.kernel_index]

        for _, parameterisation in runtime.get_parameterisations_using_support(
            self.support
        ):
            if not isinstance(parameterisation, SpatialParameterisationBasisFunction):
                continue
            del parameterisation.heights[self.kernel_index]


@dataclass(slots=True)
class EquilibriumGapBasisGrowthRefinement(IRefinementPolicy):
    """Grow Gaussian bases from the normalised EGI residual map.

    Add one EGI-seeded Gaussian after each accepted solve and retain it only
    when the configured improvement measure improves sufficiently. Combined
    FRE+EGI objectives retain their objective-cost measure; other objectives
    use the combined EGI scalar.
    """

    target: RefinementTarget
    max_basis_functions: int = 6
    relative_improvement_threshold: float = 0.05
    refinement_height_fraction: float = 0.05
    smoothing_points: int = 3
    minimum_separation_points: float = 3.0
    egi_window_weights: Sequence[float] | npt.NDArray[np.float64] | None = None
    baseline_phase_index: int | None = None
    multistart_enabled: bool = False
    multistart_offset_fraction: float = 0.10
    multistart_screening_iterations: int = 10
    fixed_basis_trajectory: bool = False
    _accepted_cost: float | None = field(default=None, init=False, repr=False)
    _resolved_egi_baseline_values: npt.NDArray[np.float64] | None = field(
        default=None, init=False, repr=False,
    )
    last_combined_egi: float | None = field(default=None, init=False)
    _accepted_spatial_parameterisations: dict[str, list[ISpatialParameterisation]] | None = field(
        default=None, init=False, repr=False,
    )

    def __post_init__(self) -> None:
        if self.max_basis_functions < 1:
            raise ValueError("max_basis_functions must be at least one.")
        if self.relative_improvement_threshold < 0.0:
            raise ValueError("relative_improvement_threshold must be non-negative.")
        if not 0.0 < self.refinement_height_fraction <= 1.0:
            raise ValueError("refinement_height_fraction must lie in (0, 1].")
        if self.smoothing_points < 1 or self.minimum_separation_points < 0.0:
            raise ValueError("smoothing_points and minimum_separation_points must be non-negative.")
        if self.baseline_phase_index is not None and self.baseline_phase_index < 0:
            raise ValueError("baseline_phase_index must be non-negative.")
        if not 0.0 < self.multistart_offset_fraction <= 1.0:
            raise ValueError("multistart_offset_fraction must lie in (0, 1].")
        if self.multistart_screening_iterations < 1:
            raise ValueError("multistart_screening_iterations must be positive.")

    def resolve_from_prior_phase(
        self,
        metrics: list[object],
        metric_results: list[object],
    ) -> None:
        """Resolve EGI baselines for objectives that do not provide them."""

        egi_results = _select_egi_metric_results(metrics, metric_results)
        self._resolved_egi_baseline_values = np.asarray(
            [
                calculate_nan_rms(extract_equilibrium_gap_temporal_rms(result))
                for result in egi_results
            ],
            dtype=np.float64,
        )

    def combine_egi_results(
        self,
        egi_results: list[object],
        objective_function: object | None,
    ) -> EquilibriumGapAggregationResult:
        """Combine EGI maps using objective-coupled or policy settings."""

        combined_objective = _global_combined_objective(objective_function)
        if combined_objective is not None:
            baselines = combined_objective.egi_baselines_for(len(egi_results))
            window_weights = combined_objective.egi_window_weights
            spatial_weights = combined_objective.egi_spatial_weights_for(
                len(egi_results)
            )
        else:
            if self._resolved_egi_baseline_values is None:
                raise ValueError(
                    "EGI basis growth baselines have not been resolved from "
                    "the configured prior phase."
                )
            baselines = self._resolved_egi_baseline_values
            window_weights = self.egi_window_weights
            spatial_weights = None
        return aggregate_equilibrium_gap_results(
            egi_results,
            egi_baseline_values=baselines,
            window_weights=window_weights,
            spatial_weights=spatial_weights,
        )

    def propose(
        self,
        runtime: PhaseRuntime,
        context: RefinementContext,
    ) -> IRefinementAction | None:
        egi_results = _evaluate_egi_metric_results(context)
        egi_aggregation = self.combine_egi_results(
            egi_results,
            context.objective_function,
        )
        self.last_combined_egi = egi_aggregation.combined_egi_spatial_rms
        if _global_combined_objective(context.objective_function) is not None:
            if context.objective_value is None or not np.isfinite(context.objective_value):
                raise ValueError("EGI basis growth requires a finite objective value.")
            cost = float(context.objective_value)
        else:
            cost = self.last_combined_egi
            if not np.isfinite(cost):
                raise ValueError("EGI basis growth requires a finite combined EGI value.")
        if self._accepted_cost is None:
            self._accept(runtime, cost)
        elif not self.fixed_basis_trajectory:
            improvement = (
                (self._accepted_cost - cost)
                / max(abs(self._accepted_cost), 1.0e-12)
            )
            if improvement < self.relative_improvement_threshold:
                assert self._accepted_spatial_parameterisations is not None
                return RestoreBasisModelAction(
                    self._accepted_spatial_parameterisations,
                )
            self._accept(runtime, cost)
        else:
            # Stage-normalised objectives may refresh references after every
            # BF addition. Their scalar values are intentionally not compared
            # across stages during a fixed-cap exploration trajectory.
            self._accept(runtime, cost)

        parameter_name, basis = _resolve_basis_parameterisation(runtime, self.target)
        return self._new_egi_basis_action(
            runtime,
            context,
            parameter_name,
            basis,
            egi_results,
            height_fraction=self.refinement_height_fraction,
        )

    def _accept(
        self,
        runtime: PhaseRuntime,
        cost: float,
    ) -> None:
        self._accepted_cost = float(cost)
        self._accepted_spatial_parameterisations = copy.deepcopy(
            runtime.spatial_parameterisations
        )

    def _new_egi_basis_action(
        self,
        runtime: PhaseRuntime,
        context: RefinementContext,
        parameter_name: str,
        basis: SpatialParameterisationBasisFunction,
        egi_results: list[object],
        *,
        height_fraction: float,
    ) -> IRefinementAction | None:
        if len(basis.kernels) >= self.max_basis_functions:
            return None
        egi_map = self.combine_egi_results(
            egi_results,
            context.objective_function,
        ).combined_baseline_scaled_egi_map
        centre = _select_egi_centre(
            egi_map,
            basis,
            context.experiment_data,
            self.smoothing_points,
            self.minimum_separation_points,
        )
        if centre is None:
            return None
        kernel, height = _default_basis(
            centre,
            basis,
            context.constitutive_parameters[parameter_name],
            height_fraction,
        )
        screening_diagnostics = None
        if self.multistart_enabled:
            kernel, height, screening_diagnostics = _screen_basis_candidates(
                runtime=runtime,
                context=context,
                parameter_name=parameter_name,
                basis=basis,
                template_kernel=kernel,
                template_height=height,
                offset_fraction=self.multistart_offset_fraction,
                screening_iterations=self.multistart_screening_iterations,
            )
        return AddBasisFunctionAction(
            basis.support,
            kernel,
            basis,
            height,
            screening_diagnostics,
        )

    def propose_initial_multistart(
        self,
        runtime: PhaseRuntime,
        context: RefinementContext,
    ) -> IRefinementAction | None:
        """Screen the EGI-seeded basis created during phase initialisation."""

        if not self.multistart_enabled:
            return None
        parameter_name, basis = _resolve_basis_parameterisation(
            runtime,
            self.target,
        )
        if len(basis.kernels) != 1 or len(basis.heights) != 1:
            return None
        height = basis.heights[0]
        if not isinstance(height, DegreeOfFreedom):
            raise TypeError("Initial multi-start basis height must be active.")
        kernel, screened_height, diagnostics = _screen_basis_candidates(
            runtime=runtime,
            context=context,
            parameter_name=parameter_name,
            basis=basis,
            template_kernel=basis.kernels[0],
            template_height=height,
            offset_fraction=self.multistart_offset_fraction,
            screening_iterations=self.multistart_screening_iterations,
        )
        return ReplaceBasisFunctionAction(
            support=basis.support,
            kernel_index=0,
            kernel=kernel,
            seed_parameterisation=basis,
            seed_height=screened_height,
            screening_diagnostics=diagnostics,
        )


@dataclass(slots=True)
class SensitivityCorrectionBasisGrowthRefinement(
    EquilibriumGapBasisGrowthRefinement
):
    """Grow a Gaussian from the objective's predicted material correction."""

    sensitivity_perturbation_factor: float = 0.01
    correction_feature_fraction: float = 0.2
    last_correction_map: npt.NDArray[np.float64] | None = field(
        default=None,
        init=False,
        repr=False,
    )

    def __post_init__(self) -> None:
        EquilibriumGapBasisGrowthRefinement.__post_init__(self)
        if not 0.0 < self.sensitivity_perturbation_factor < 1.0:
            raise ValueError(
                "sensitivity_perturbation_factor must lie in (0, 1)."
            )
        if not 0.0 < self.correction_feature_fraction <= 1.0:
            raise ValueError("correction_feature_fraction must lie in (0, 1].")

    def _new_egi_basis_action(
        self,
        runtime: PhaseRuntime,
        context: RefinementContext,
        parameter_name: str,
        basis: SpatialParameterisationBasisFunction,
        egi_results: list[object],
        *,
        height_fraction: float,
    ) -> IRefinementAction | None:
        _ = egi_results
        if len(basis.kernels) >= self.max_basis_functions:
            return None
        objective = _global_combined_objective(context.objective_function)
        if objective is None:
            raise TypeError(
                "Sensitivity-correction growth requires "
                "CombinedForceAndEquilibriumGapObjective, or an objective "
                "which wraps one as global_objective."
            )

        egi_metrics: list[EquilibriumGapMetric] = []
        metric_results = []
        force_metric: SliceWiseForceReconstructionMetric | None = None
        for metric in context.metrics:
            if isinstance(metric, EquilibriumGapMetric):
                egi_metrics.append(metric)
                metric_results.append(
                    metric.evaluate_equilibrium_gap(context.stress).metric_result
                )
            elif isinstance(metric, SliceWiseForceReconstructionMetric):
                if force_metric is not None:
                    raise ValueError(
                        "Sensitivity-correction growth requires one FRE metric."
                    )
                force_metric = metric
                metric_results.append(
                    metric.evaluate_force_recon_error(
                        context.stress,
                        context.experiment_data,
                    ).metric_result
                )
        if force_metric is None:
            raise ValueError(
                "Sensitivity-correction growth requires one FRE metric."
            )

        cotangents = objective.residual_cotangents(metric_results)
        stress_gradient = force_metric.normalised_residual_stress_adjoint(
            cotangents.force,
            context.experiment_data,
        )
        for metric, cotangent in zip(
            egi_metrics,
            cotangents.equilibrium_gap,
            strict=True,
        ):
            stress_gradient += metric.normalised_gap_stress_adjoint(
                context.stress,
                cotangent,
            )

        local_stress_sensitivity = calculate_local_parameter_stress_sensitivity(
            context.experiment_data.strain,
            context.stress,
            context.constitutive_law,
            context.parameter_maps,
            parameter_name,
            self.sensitivity_perturbation_factor,
        )
        correction = -np.nansum(
            stress_gradient * local_stress_sensitivity,
            axis=(0, 1),
        )
        self.last_correction_map = correction.copy()
        proposal = _basis_from_correction_map(
            correction,
            basis,
            context.experiment_data,
            context.constitutive_parameters[parameter_name],
            height_fraction=height_fraction,
            smoothing_points=self.smoothing_points,
            minimum_separation_points=self.minimum_separation_points,
            feature_fraction=self.correction_feature_fraction,
        )
        if proposal is None:
            return None
        kernel, height, diagnostics = proposal
        if context.objective_function is not objective:
            diagnostics["correction_cotangent_source"] = (
                "wrapped_global_mechanical_closure"
            )
        parameter_direction = (
            height.value * basis._evaluate_kernel_response(kernel)
        )
        directional_derivative = float(
            np.sum(-correction * parameter_direction)
        )
        diagnostics["predicted_objective_directional_derivative"] = (
            directional_derivative
        )
        if directional_derivative >= 0.0:
            return None
        return AddBasisFunctionAction(
            basis.support,
            kernel,
            basis,
            height,
            diagnostics,
        )

@dataclass(slots=True)
class FitBasisFunctionsToMapAction(IRefinementAction):
    basis: SpatialParameterisationBasisFunction
    target_map: npt.NDArray[np.float64]
    parameter_range: float
    max_basis_functions: int
    minimum_relative_improvement: float

    def apply(self, runtime: PhaseRuntime, context: RefinementContext) -> None:
        _ = runtime, context
        self.basis.fit_to_map(
            self.target_map,
            parameter_range=self.parameter_range,
            max_basis_functions=self.max_basis_functions,
            minimum_relative_improvement=self.minimum_relative_improvement,
        )


@dataclass(slots=True)
class RestoreBasisModelAction(IRefinementAction):
    spatial_parameterisations: dict[str, list[ISpatialParameterisation]]

    @property
    def terminal(self) -> bool:
        return True

    @property
    def accepts_current_solve(self) -> bool:
        return False

    def apply(self, runtime: PhaseRuntime, context: RefinementContext) -> None:
        _ = context
        runtime.adopt_spatial_parameterisations(
            copy.deepcopy(self.spatial_parameterisations)
        )


def _global_combined_objective(
    objective_function: object | None,
) -> CombinedForceAndEquilibriumGapObjective | None:
    """Return the closure objective used by EGI-based refinement.

    A material-information objective is deliberately differentiated through
    its frozen global mechanical-closure component for basis placement.  Its
    tail/coherence reductions do not yet expose stress adjoints, so pretending
    otherwise would make the sensitivity-correction direction inconsistent.
    """

    if isinstance(objective_function, CombinedForceAndEquilibriumGapObjective):
        return objective_function
    wrapped = getattr(objective_function, "global_objective", None)
    if isinstance(wrapped, CombinedForceAndEquilibriumGapObjective):
        return wrapped
    return None


def _resolve_basis_parameterisation(
    runtime: PhaseRuntime,
    target: RefinementTarget,
) -> tuple[str, SpatialParameterisationBasisFunction]:
    if isinstance(target, tuple):
        parameter_name, parameterisation = runtime.get_parameterisation(*target)
    else:
        matches = [
            (parameter_name, parameterisation)
            for parameter_name, parameterisations in runtime.spatial_parameterisations.items()
            for parameterisation in parameterisations
            if parameterisation is target
        ]
        if len(matches) != 1:
            raise TypeError(
                "EGI basis growth target must identify one phase parameterisation."
            )
        parameter_name, parameterisation = matches[0]
    if not isinstance(parameterisation, SpatialParameterisationBasisFunction):
        raise TypeError("EGI basis growth target must be basis-based.")
    return parameter_name, parameterisation


def _map_rms(values: npt.NDArray[np.float64]) -> float:
    finite = values[np.isfinite(values)]
    return 0.0 if finite.size == 0 else float(np.sqrt(np.mean(finite**2)))


def _evaluate_egi_metric_results(
    context: RefinementContext,
) -> list[object]:
    results = []
    for metric in context.metrics:
        if isinstance(metric, EquilibriumGapMetric):
            results.append(metric.evaluate_equilibrium_gap(context.stress).metric_result)
    if not results:
        raise ValueError("EGI basis growth requires at least one EquilibriumGapMetric.")
    return results


def _select_egi_metric_results(
    metrics: list[object],
    metric_results: list[object],
) -> list[object]:
    if len(metrics) != len(metric_results):
        raise ValueError("Metric results do not match the phase metric definitions.")
    results = [
        result
        for metric, result in zip(metrics, metric_results, strict=True)
        if isinstance(metric, EquilibriumGapMetric)
    ]
    if not results:
        raise ValueError("EGI basis growth requires at least one EquilibriumGapMetric.")
    return results


def _select_egi_centre(
    egi_map: npt.NDArray[np.float64],
    basis: SpatialParameterisationBasisFunction,
    experiment_data: ExperimentData,
    smoothing_points: int,
    minimum_separation_points: float,
) -> tuple[float, float] | None:
    x = experiment_data.specimen_geometry.x
    y = experiment_data.specimen_geometry.y
    specimen = experiment_data.specimen_geometry.region_of_interest.sample_specimen_mask(x, y)
    smoothed = uniform_filter(np.where(np.isfinite(egi_map), egi_map, 0.0), size=smoothing_points)
    support = uniform_filter(np.isfinite(egi_map).astype(float), size=smoothing_points)
    candidates = np.where(specimen & (support > 0.0), smoothed, np.nan)
    spacing = min(
        float(np.nanmedian(np.diff(x, axis=1))),
        float(np.nanmedian(np.diff(y, axis=0))),
    )
    minimum_separation = minimum_separation_points * spacing
    for index in np.argsort(np.nan_to_num(candidates, nan=-np.inf).ravel())[::-1]:
        row, column = np.unravel_index(index, candidates.shape)
        if not np.isfinite(candidates[row, column]):
            continue
        centre = (float(x[row, column]), float(y[row, column]))
        if all(
            np.hypot(centre[0] - _kernel_value(kernel.x), centre[1] - _kernel_value(kernel.y))
            >= minimum_separation
            for kernel in basis.kernels
        ):
            return centre
    return None


def _basis_from_correction_map(
    correction: npt.NDArray[np.float64],
    basis: SpatialParameterisationBasisFunction,
    experiment_data: ExperimentData,
    parameter: ConstitutiveParameter,
    *,
    height_fraction: float,
    smoothing_points: int,
    minimum_separation_points: float,
    feature_fraction: float,
) -> tuple[BasisFunctionKernel, DegreeOfFreedom, dict[str, object]] | None:
    """Fit one continuous Gaussian seed to the dominant signed correction."""
    x = experiment_data.specimen_geometry.x
    y = experiment_data.specimen_geometry.y
    specimen = (
        experiment_data.specimen_geometry.region_of_interest.sample_specimen_mask(
            x, y
        )
    )
    finite = specimen & np.isfinite(correction)
    support = uniform_filter(finite.astype(float), size=smoothing_points)
    smoothed = np.divide(
        uniform_filter(np.where(finite, correction, 0.0), size=smoothing_points),
        support,
        out=np.zeros_like(correction),
        where=support > 0.0,
    )
    candidates = np.where(finite & (support > 0.0), np.abs(smoothed), -np.inf)
    spacing = min(
        abs(float(np.nanmedian(np.diff(x, axis=1)))),
        abs(float(np.nanmedian(np.diff(y, axis=0)))),
    )
    minimum_separation = minimum_separation_points * spacing
    peak_index: tuple[int, int] | None = None
    for flat_index in np.argsort(candidates.ravel())[::-1]:
        index = np.unravel_index(flat_index, candidates.shape)
        if not np.isfinite(candidates[index]) or candidates[index] <= 0.0:
            break
        centre = (float(x[index]), float(y[index]))
        if all(
            np.hypot(
                centre[0] - _kernel_value(kernel.x),
                centre[1] - _kernel_value(kernel.y),
            ) >= minimum_separation
            for kernel in basis.kernels
        ):
            peak_index = index
            break
    if peak_index is None:
        return None

    peak_value = float(smoothed[peak_index])
    feature_mask = (
        finite
        & (np.sign(smoothed) == np.sign(peak_value))
        & (np.abs(smoothed) >= feature_fraction * abs(peak_value))
    )
    components, _ = label(feature_mask)
    feature_mask &= components == components[peak_index]
    weights = np.where(feature_mask, np.abs(smoothed), 0.0)
    weight_sum = float(np.sum(weights))
    if weight_sum <= 0.0:
        return None
    centre_x = float(np.sum(weights * x) / weight_sum)
    centre_y = float(np.sum(weights * y) / weight_sum)
    dx = x - centre_x
    dy = y - centre_y
    covariance = np.array(
        [
            [np.sum(weights * dx * dx), np.sum(weights * dx * dy)],
            [np.sum(weights * dx * dy), np.sum(weights * dy * dy)],
        ],
        dtype=np.float64,
    ) / weight_sum
    covariance += np.eye(2) * spacing**2

    min_x, max_x, min_y, max_y = basis.get_centre_bounds()
    kernel = basis.create_kernel_from_covariance(
        DegreeOfFreedom(centre_x, min_x, max_x),
        DegreeOfFreedom(centre_y, min_y, max_y),
        covariance,
    )
    parameter_span = parameter.upper_bound - parameter.lower_bound
    height = DegreeOfFreedom(
        float(np.sign(peak_value) * height_fraction * parameter_span),
        -parameter_span,
        parameter_span,
    )
    diagnostics: dict[str, object] = {
        "policy": "sensitivity_correction",
        "peak_correction_gradient": peak_value,
        "proposed_sign": float(np.sign(peak_value)),
        "centre": [centre_x, centre_y],
        "covariance": covariance.tolist(),
        "feature_point_count": int(np.sum(feature_mask)),
        "correction_l2": float(np.sqrt(np.sum(correction[finite] ** 2))),
    }
    return kernel, height, diagnostics


def _kernel_value(value: float | DegreeOfFreedom) -> float:
    return float(value.value if isinstance(value, DegreeOfFreedom) else value)


def _default_basis(
    centre: tuple[float, float],
    basis: SpatialParameterisationBasisFunction,
    parameter: ConstitutiveParameter,
    height_fraction: float,
) -> tuple[BasisFunctionKernel, DegreeOfFreedom]:
    x, y = basis.x, basis.y
    spacing = min(
        float(np.nanmedian(np.diff(x, axis=1))),
        float(np.nanmedian(np.diff(y, axis=0))),
    )
    diagonal = float(np.hypot(np.nanmax(x) - np.nanmin(x), np.nanmax(y) - np.nanmin(y)))
    span = parameter.upper_bound - parameter.lower_bound
    minimum_variance = (3.0 * spacing) ** 2
    maximum_variance = max(diagonal**2, minimum_variance * (1.0 + 1.0e-6))
    initial_variance = float(np.sqrt(minimum_variance * maximum_variance))
    min_x, max_x, min_y, max_y = basis.get_centre_bounds()
    return (
        basis.create_kernel(
            DegreeOfFreedom(centre[0], min_x, max_x),
            DegreeOfFreedom(centre[1], min_y, max_y),
            DegreeOfFreedom(
                initial_variance,
                minimum_variance,
                maximum_variance,
                scaling="log",
            ),
        ),
        DegreeOfFreedom(height_fraction * span, -span, span),
    )


def _screen_basis_candidates(
    *,
    runtime: PhaseRuntime,
    context: RefinementContext,
    parameter_name: str,
    basis: SpatialParameterisationBasisFunction,
    template_kernel: BasisFunctionKernel,
    template_height: DegreeOfFreedom,
    offset_fraction: float,
    screening_iterations: int,
) -> tuple[BasisFunctionKernel, DegreeOfFreedom, dict[str, object]]:
    """Screen five centre seeds while only the proposed basis remains active."""

    if not isinstance(runtime.optimiser, OptimiserPatternSearch):
        raise TypeError(
            "Multi-start basis screening requires OptimiserPatternSearch."
        )
    if context.objective_function is None:
        raise ValueError("Multi-start basis screening requires an objective.")

    min_x, max_x, min_y, max_y = basis.get_centre_bounds()
    raw_centre = (
        _kernel_value(template_kernel.x),
        _kernel_value(template_kernel.y),
    )
    delta_x = offset_fraction * (max_x - min_x)
    delta_y = offset_fraction * (max_y - min_y)
    candidate_specs = (
        ("peak", raw_centre),
        ("x_minus", (raw_centre[0] - delta_x, raw_centre[1])),
        ("x_plus", (raw_centre[0] + delta_x, raw_centre[1])),
        ("y_minus", (raw_centre[0], raw_centre[1] - delta_y)),
        ("y_plus", (raw_centre[0], raw_centre[1] + delta_y)),
    )

    screened: list[dict[str, object]] = []
    winners: list[tuple[BasisFunctionKernel, DegreeOfFreedom]] = []
    for label, proposed_centre in candidate_specs:
        start_centre = (
            float(np.clip(proposed_centre[0], min_x, max_x)),
            float(np.clip(proposed_centre[1], min_y, max_y)),
        )
        candidate_kernel = copy.deepcopy(template_kernel)
        _set_kernel_centre(candidate_kernel, start_centre)
        candidate_height = copy.deepcopy(template_height)
        screening_model = _build_screening_model(
            context,
            parameter_name,
            basis,
            candidate_kernel,
            candidate_height,
        )
        number_of_dofs = PhaseSpatialState(
            screening_model
        ).get_num_degrees_of_freedom()
        max_evaluations = (
            1 + screening_iterations * (2 * number_of_dofs + 1)
        )
        optimiser = _screening_optimiser(
            runtime.optimiser,
            screening_iterations,
            max_evaluations,
        )
        outcome = optimiser.optimise(
            context.constitutive_law,
            context.parameter_map_size,
            screening_model,
            [copy.copy(metric) for metric in context.metrics],
            copy.deepcopy(context.objective_function),
            context.experiment_data,
        )
        if outcome.solve_result is None:
            raise RuntimeError("Multi-start screening did not return diagnostics.")
        optimised_basis = outcome.spatial_parameterisations[parameter_name][-1]
        if not isinstance(
            optimised_basis,
            SpatialParameterisationBasisFunction,
        ):
            raise TypeError("Multi-start screening lost the candidate basis.")
        optimised_kernel = copy.deepcopy(optimised_basis.kernels[0])
        optimised_height = optimised_basis.heights[0]
        if not isinstance(optimised_height, DegreeOfFreedom):
            raise TypeError("Screened basis height must remain active.")
        refined_centre = (
            _kernel_value(optimised_kernel.x),
            _kernel_value(optimised_kernel.y),
        )
        solve = outcome.solve_result
        screened.append({
            "label": label,
            "proposed_position": [
                float(proposed_centre[0]),
                float(proposed_centre[1]),
            ],
            "starting_position": [start_centre[0], start_centre[1]],
            "refined_position": [refined_centre[0], refined_centre[1]],
            "cost": float(solve.final_objective["cost"]),
            "evaluations": int(solve.num_evaluations),
            "iterations": int(solve.final_objective.get("iterations", 0)),
            "status": solve.status,
            "started_on_bound": bool(
                np.isclose(start_centre[0], min_x)
                or np.isclose(start_centre[0], max_x)
                or np.isclose(start_centre[1], min_y)
                or np.isclose(start_centre[1], max_y)
            ),
            "refined_on_bound": bool(
                np.isclose(refined_centre[0], min_x)
                or np.isclose(refined_centre[0], max_x)
                or np.isclose(refined_centre[1], min_y)
                or np.isclose(refined_centre[1], max_y)
            ),
        })
        winners.append((optimised_kernel, copy.deepcopy(optimised_height)))

    selected_index = int(np.argmin([float(entry["cost"]) for entry in screened]))
    selected_kernel, selected_height = winners[selected_index]
    diagnostics: dict[str, object] = {
        "enabled": True,
        "offset_fraction": float(offset_fraction),
        "screening_iterations": int(screening_iterations),
        "raw_egi_peak": [raw_centre[0], raw_centre[1]],
        "offset_distance": [float(delta_x), float(delta_y)],
        "selected_candidate_index": selected_index,
        "selected_candidate_label": screened[selected_index]["label"],
        "selected_cost": screened[selected_index]["cost"],
        "candidates": screened,
    }
    return selected_kernel, selected_height, diagnostics


def _build_screening_model(
    context: RefinementContext,
    parameter_name: str,
    basis: SpatialParameterisationBasisFunction,
    kernel: BasisFunctionKernel,
    height: DegreeOfFreedom,
) -> dict[str, list[ISpatialParameterisation]]:
    screening_model: dict[str, list[ISpatialParameterisation]] = {
        name: [SpatialParameterisationKnown(np.asarray(values).copy())]
        for name, values in context.parameter_maps.items()
    }
    candidate_basis = SpatialParameterisationBasisFunction(
        x=basis.x,
        y=basis.y,
        kernels=[copy.deepcopy(kernel)],
        heights=[copy.deepcopy(height)],
        kernel_type=basis.kernel_type,
        centre_bounds_span_factor=basis.centre_bounds_span_factor,
    )
    screening_model[parameter_name].append(candidate_basis)
    return screening_model


def _screening_optimiser(
    source: OptimiserPatternSearch,
    iterations: int,
    max_evaluations: int,
) -> OptimiserPatternSearch:
    return OptimiserPatternSearch(
        initial_mesh_size=source.initial_mesh_size,
        minimum_mesh_size=source.minimum_mesh_size,
        mesh_contraction_factor=source.mesh_contraction_factor,
        mesh_expansion_factor=source.mesh_expansion_factor,
        pattern_step_size=source.pattern_step_size,
        max_iterations=iterations,
        max_evaluations=max_evaluations,
        objective_absolute_tolerance=source.objective_absolute_tolerance,
        objective_relative_tolerance=source.objective_relative_tolerance,
        parallel_workers=source.parallel_workers,
        random_seed=source.random_seed,
        max_batch_size=source.max_batch_size,
    )


def _set_kernel_centre(
    kernel: BasisFunctionKernel,
    centre: tuple[float, float],
) -> None:
    if not isinstance(kernel.x, DegreeOfFreedom) or not isinstance(
        kernel.y,
        DegreeOfFreedom,
    ):
        raise TypeError("Multi-start basis centres must be active DOFs.")
    kernel.x.value = float(centre[0])
    kernel.y.value = float(centre[1])


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


def _resolve_basis_seed_parameterisation(
    runtime: PhaseRuntime,
    support: SupportBasis,
    target: RefinementTarget,
    seed_parameter_name: str | None,
) -> tuple[str, SpatialParameterisationBasisFunction]:
    if isinstance(target, tuple):
        parameter_name, parameterisation = runtime.get_parameterisation(
            target[0],
            target[1],
        )
        if not isinstance(parameterisation, SpatialParameterisationBasisFunction):
            raise TypeError(
                "Basis refinement parameterisation target must be basis-based."
            )
        return parameter_name, parameterisation

    for parameter_name, parameterisation in runtime.get_parameterisations_using_support(
        support
    ):
        if seed_parameter_name is not None and parameter_name != seed_parameter_name:
            continue
        if isinstance(parameterisation, SpatialParameterisationBasisFunction):
            return parameter_name, parameterisation

    raise ValueError("No basis parameterisation uses the targeted support.")


def _find_removable_basis_index(
    runtime: PhaseRuntime,
    support: SupportBasis,
    remove_height_threshold: float | None,
    min_basis_functions: int,
) -> int | None:
    if remove_height_threshold is None:
        return None

    assert support.kernels is not None
    if len(support.kernels) <= min_basis_functions:
        return None

    candidate_scores = np.zeros(len(support.kernels), dtype=np.float64)
    for _, parameterisation in runtime.get_parameterisations_using_support(support):
        if not isinstance(parameterisation, SpatialParameterisationBasisFunction):
            continue
        for index, height in enumerate(parameterisation.heights):
            value = getattr(height, "value", height)
            if value is None:
                value = np.inf
            candidate_scores[index] = max(candidate_scores[index], abs(float(value)))

    removable_indices = np.flatnonzero(candidate_scores <= remove_height_threshold)
    if removable_indices.size == 0:
        return None
    return int(removable_indices[0])
