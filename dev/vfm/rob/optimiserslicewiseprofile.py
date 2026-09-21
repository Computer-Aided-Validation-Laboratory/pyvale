"""Experimental initial-guess-independent profiling of scalar slice problems."""

from __future__ import annotations

import copy
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import numpy.typing as npt

from pyvale.vfm.constlaw import IConstitutiveLaw
from pyvale.vfm.experimentdata import ExperimentData
from pyvale.vfm.identificationresult import OptimisationOutcome, SolveResult, snapshot_object
from pyvale.vfm.metric import IMetric
from pyvale.vfm.metricsliceforce import SliceWiseForceReconstructionMetric
from pyvale.vfm.objectivefunc import IObjectiveFunction, IVectorObjectiveFunction
from pyvale.vfm.optimiser import IOptimiser
from optimiserscalarprofile import solve_scalar_profile
from pyvale.vfm.optimiserslicewiseindependent import (
    _build_slice_solve_data,
    _collect_spatial_dof_values,
    _evaluate_slice_candidate,
    _prepare_slice_constitutive_law,
    _update_slice_parameterisations,
    _validate_slice_parameterisations,
)
from pyvale.vfm.spatialparam import ISpatialParameterisation


class SliceWiseIndependentProfileOptimiser(IOptimiser):
    """Solve exactly one unknown per slice using a scan and bounded refinements.

    Every endpoint and every distinct bracketed scan minimum is retained as a
    candidate.  The complete cost profile and numerical evidence classification
    are stored in each child ``SolveResult.details``.
    """

    def __init__(
        self,
        *,
        scan_points: int = 31,
        relative_cost_tolerance: float = 0.01,
        absolute_cost_tolerance: float = 1.0e-12,
        parallel_workers: int = 1,
    ) -> None:
        if scan_points < 5:
            raise ValueError("scan_points must be at least five")
        if isinstance(parallel_workers, bool) or not isinstance(parallel_workers, (int, np.integer)) or parallel_workers < 1:
            raise ValueError("parallel_workers must be a positive integer")
        self.scan_points = int(scan_points)
        self.relative_cost_tolerance = float(relative_cost_tolerance)
        self.absolute_cost_tolerance = float(absolute_cost_tolerance)
        self.parallel_workers = int(parallel_workers)

    def get_required_objective_function_type(self) -> type:
        return IVectorObjectiveFunction

    def optimise(
        self,
        constitutive_law: IConstitutiveLaw,
        parameter_map_size: npt.NDArray[np.uint32],
        spatial_parameterisations: dict[str, list[ISpatialParameterisation]],
        metrics: list[IMetric],
        objective_function: IObjectiveFunction,
        experiment_data: ExperimentData,
        progress_callback=None,
    ) -> OptimisationOutcome:
        del progress_callback
        if len(metrics) != 1 or not isinstance(metrics[0], SliceWiseForceReconstructionMetric):
            raise ValueError(
                "SliceWiseIndependentProfileOptimiser requires exactly one "
                "SliceWiseForceReconstructionMetric."
            )
        metric = metrics[0]
        if metric.slice_partition is None:
            raise RuntimeError("Slice metric partition must be prepared before optimisation.")
        _validate_slice_parameterisations(spatial_parameterisations, metric)

        optimised = copy.deepcopy(spatial_parameterisations)
        started = time.perf_counter()

        def solve_slice(slice_index: int):
            local = _build_slice_solve_data(
                slice_index=slice_index,
                parameter_map_size=parameter_map_size,
                spatial_parameterisations=spatial_parameterisations,
                slice_metric=metric,
                experiment_data=experiment_data,
            )
            if not local.unknown_parameter_names:
                return None
            if len(local.unknown_parameter_names) != 1:
                raise ValueError(
                    "SliceWiseIndependentProfileOptimiser requires exactly one "
                    f"unknown per active slice; slice {slice_index} has "
                    f"{len(local.unknown_parameter_names)}."
                )
            if local.local_point_indices.size == 0:
                raise ValueError(f"Slice {slice_index} has no usable points.")
            local_law = _prepare_slice_constitutive_law(constitutive_law, local)
            lower = float(local.lower_bounds[0])
            upper = float(local.upper_bounds[0])

            def residual(value: float) -> npt.NDArray[np.float64]:
                normalised = np.asarray([(value - lower) / (upper - lower)])
                return _evaluate_slice_candidate(
                    normalised,
                    local_law,
                    objective_function,
                    metric,
                    experiment_data,
                    local,
                )

            slice_started = time.perf_counter()
            profile = solve_scalar_profile(
                residual,
                (lower, upper),
                scan_points=self.scan_points,
                relative_cost_tolerance=self.relative_cost_tolerance,
                absolute_cost_tolerance=self.absolute_cost_tolerance,
            )
            return local, profile, time.perf_counter() - slice_started

        executor = None
        outputs = None
        if self.parallel_workers == 1:
            outputs = map(solve_slice, range(metric.slice_partition.num_slices))
        else:
            executor = ThreadPoolExecutor(max_workers=self.parallel_workers)
            outputs = executor.map(solve_slice, range(metric.slice_partition.num_slices))

        children: list[SolveResult] = []
        skipped = 0
        try:
            for slice_index, output in enumerate(outputs):
                if output is None:
                    skipped += 1
                    continue
                local, profile, runtime = output
                _update_slice_parameterisations(
                    optimised,
                    slice_index,
                    local.unknown_parameter_names,
                    np.asarray([profile.optimum]),
                )
                summary = profile.to_summary()
                children.append(
                    SolveResult(
                        solve_iteration=slice_index,
                        optimiser=snapshot_object(self, options=self._options()),
                        runtime_seconds=runtime,
                        num_evaluations=profile.evaluation_count,
                        success=True,
                        status=profile.classification,
                        message="Deterministic scan plus all bracketed scalar refinements.",
                        initial_dofs=[],
                        final_dofs=[profile.optimum],
                        final_objective={
                            "cost": profile.minimum_cost,
                            "residual_norm": float(np.linalg.norm(profile.residual)),
                            "residual_size": int(profile.residual.size),
                            "finite_residual_count": int(np.count_nonzero(np.isfinite(profile.residual))),
                        },
                        details={
                            "slice_index": slice_index,
                            "unknown_parameter_names": list(local.unknown_parameter_names),
                            "num_local_points": int(local.global_point_indices.size),
                            **summary,
                        },
                    )
                )
        finally:
            if executor is not None:
                executor.shutdown()

        classifications: dict[str, int] = {}
        for child in children:
            classifications[str(child.status)] = classifications.get(str(child.status), 0) + 1
        return OptimisationOutcome(
            spatial_parameterisations=optimised,
            solve_result=SolveResult(
                solve_iteration=0,
                optimiser=snapshot_object(self, options=self._options()),
                runtime_seconds=time.perf_counter() - started,
                num_evaluations=sum(child.num_evaluations or 0 for child in children),
                success=True,
                status="completed",
                message="Profiled every active scalar slice independently.",
                initial_dofs=_collect_spatial_dof_values(spatial_parameterisations),
                final_dofs=_collect_spatial_dof_values(optimised),
                details={
                    "num_slices": metric.slice_partition.num_slices,
                    "solved_slice_count": len(children),
                    "skipped_slice_count": skipped,
                    "classifications": classifications,
                    **self._options(),
                },
                children=children,
            ),
        )

    def _options(self) -> dict[str, int | float]:
        return {
            "scan_points": self.scan_points,
            "relative_cost_tolerance": self.relative_cost_tolerance,
            "absolute_cost_tolerance": self.absolute_cost_tolerance,
            "parallel_workers": self.parallel_workers,
        }
