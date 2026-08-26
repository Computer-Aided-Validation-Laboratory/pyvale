"""Parallel generalized pattern search for bounded scalar VFM objectives.

The search operates in normalised parameter space, polls complete positive and
negative direction sets, and evaluates independent candidates concurrently.
It alternates coordinate and seeded orthonormal bases to explore both direct
and correlated parameter changes.
"""

from __future__ import annotations

import copy
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, is_dataclass
import threading
import time

import numpy as np
import numpy.typing as npt

from pyvale.vfm.constlaw import IConstitutiveLaw
from pyvale.vfm.experimentdata import ExperimentData
from pyvale.vfm.identificationresult import OptimisationOutcome, SolveResult, snapshot_object
from pyvale.vfm.metric import IMetric
from pyvale.vfm.objectivefunc import IObjectiveFunction, IScalarObjectiveFunction
from pyvale.vfm.optimiser import IOptimiser, evaluate_candidate
from pyvale.vfm.progress import ProgressEvent, emit_progress
from pyvale.vfm.spatialparam import ISpatialParameterisation, PhaseSpatialState


class OptimiserPatternSearch(IOptimiser):
    """Bounded generalized pattern search with parallel complete polling.

    All calculations occur in normalised DOF space. Iterations alternate
    coordinate and seeded random-orthonormal direction bases, evaluate the
    positive and negative directions as a complete poll, and accept the best
    improving candidate. A successful displacement also supplies a
    Hooke--Jeeves pattern candidate for the next iteration.

    Poll candidates are independent and may be evaluated concurrently. By
    default, the mesh is held constant after improvement and contracted only
    after an unsuccessful complete poll. An expansion factor greater than one
    enables explicit expansion after successful iterations.
    """

    def __init__(self, *, initial_mesh_size: float = 0.1,
                 minimum_mesh_size: float = 1.0e-3,
                 mesh_contraction_factor: float = 0.5,
                 mesh_expansion_factor: float = 1.0,
                 pattern_step_size: float = 1.0,
                 max_iterations: int = 100,
                 max_evaluations: int = 1000,
                 objective_absolute_tolerance: float = 0.0,
                 objective_relative_tolerance: float = 1.0e-12,
                 parallel_workers: int = 1,
                 random_seed: int = 0,
                 max_batch_size: int | None = None) -> None:
        """Configure the bounded pattern search.

        Args:
            initial_mesh_size: Initial poll distance in normalised DOF space.
            minimum_mesh_size: Mesh size at which the search terminates.
            mesh_contraction_factor: Multiplier applied after a failed poll.
            mesh_expansion_factor: Multiplier applied after a successful poll;
                one holds the mesh size constant.
            pattern_step_size: Extrapolation multiplier for the previous
                successful displacement; zero disables pattern candidates.
            max_iterations: Maximum number of complete poll iterations.
            max_evaluations: Maximum number of cached objective evaluations.
            objective_absolute_tolerance: Absolute part of the improvement
                threshold.
            objective_relative_tolerance: Scale-relative part of the
                improvement threshold.
            parallel_workers: Number of objective-evaluation threads.
            random_seed: Seed used to generate reproducible orthonormal bases.
            max_batch_size: Maximum candidates submitted in one batch, or
                ``None`` to submit the complete set together.
        """
        if not 0.0 < minimum_mesh_size <= initial_mesh_size <= 1.0:
            raise ValueError("Require 0 < minimum_mesh_size <= initial_mesh_size <= 1.")
        if not 0.0 < mesh_contraction_factor < 1.0:
            raise ValueError("mesh_contraction_factor must lie in (0, 1).")
        if mesh_expansion_factor < 1.0:
            raise ValueError("mesh_expansion_factor must be at least one.")
        if pattern_step_size < 0.0:
            raise ValueError("pattern_step_size must be non-negative.")
        if max_iterations < 1 or max_evaluations < 1:
            raise ValueError("Iteration and evaluation limits must be positive.")
        if objective_absolute_tolerance < 0.0 or objective_relative_tolerance < 0.0:
            raise ValueError("Objective tolerances must be non-negative.")
        if parallel_workers < 1:
            raise ValueError("parallel_workers must be positive.")
        if max_batch_size is not None and max_batch_size < 1:
            raise ValueError("max_batch_size must be positive or None.")
        self.initial_mesh_size = float(initial_mesh_size)
        self.minimum_mesh_size = float(minimum_mesh_size)
        self.mesh_contraction_factor = float(mesh_contraction_factor)
        self.mesh_expansion_factor = float(mesh_expansion_factor)
        self.pattern_step_size = float(pattern_step_size)
        self.max_iterations = int(max_iterations)
        self.max_evaluations = int(max_evaluations)
        self.objective_absolute_tolerance = float(objective_absolute_tolerance)
        self.objective_relative_tolerance = float(objective_relative_tolerance)
        self.parallel_workers = int(parallel_workers)
        self.random_seed = int(random_seed)
        self.max_batch_size = (
            None if max_batch_size is None else int(max_batch_size)
        )

    def get_required_objective_function_type(self) -> type:
        """Return the scalar objective interface required by this optimiser."""
        return IScalarObjectiveFunction

    def optimise(self, constitutive_law: IConstitutiveLaw,
                 parameter_map_size: npt.NDArray[np.uint32],
                 spatial_parameterisations: dict[str, list[ISpatialParameterisation]],
                 metrics: list[IMetric], objective_function: IObjectiveFunction,
                 experiment_data: ExperimentData, progress_callback=None) -> OptimisationOutcome:
        """Minimise a bounded scalar objective using parallel complete polls.

        The returned outcome contains the accepted physical parameter values,
        stopping condition, evaluation count, and per-iteration search history.
        """
        # Work in [0, 1] so one mesh size is meaningful across physical DOFs
        # with different units and bounds.
        state = PhaseSpatialState(spatial_parameterisations)
        initial_dofs = state.collect_degrees_of_freedom()
        x = np.clip(state.collect_normalised_degrees_of_freedom(), 0.0, 1.0)
        if x.size == 0:
            return _empty_outcome(spatial_parameterisations, self)

        # Cache normalised points to avoid repeating expensive VFM evaluations.
        cache: dict[tuple[float, ...], float] = {}
        evaluations = 0
        worker_local = threading.local()

        def checked_cost(value: float) -> float:
            cost = float(value)
            if not np.isfinite(cost):
                raise ValueError(
                    "Pattern-search objective returned a non-finite value."
                )
            return cost

        def evaluate(candidate: npt.NDArray[np.float64]) -> float:
            nonlocal evaluations
            bounded = np.clip(candidate, 0.0, 1.0)
            key = tuple(np.round(bounded, 14))
            if key not in cache:
                cache[key] = checked_cost(evaluate_candidate(
                    bounded, constitutive_law, parameter_map_size, state,
                    metrics, objective_function, experiment_data,
                ))
                evaluations += 1
                emit_progress(progress_callback, ProgressEvent(
                    message=f"pattern-search evaluation {evaluations}: cost={cache[key]:.8g}",
                    kind="evaluation",
                ))
            return cache[key]

        def evaluate_in_worker(candidate: npt.NDArray[np.float64]) -> float:
            # Each thread owns its mutable metric and objective diagnostics.
            context = getattr(worker_local, "context", None)
            if context is None:
                local_metrics = []
                for metric in metrics:
                    local_metric = copy.copy(metric)
                    if hasattr(local_metric, "_kernel_fft_cache"):
                        local_metric._kernel_fft_cache = dict(
                            local_metric._kernel_fft_cache
                        )
                    local_metrics.append(local_metric)
                context = (
                    local_metrics,
                    copy.deepcopy(objective_function),
                )
                worker_local.context = context
            local_metrics, local_objective = context
            return checked_cost(evaluate_candidate(
                candidate, constitutive_law, parameter_map_size, state,
                local_metrics, local_objective, experiment_data,
            ))

        def evaluate_poll(
            candidates: list[npt.NDArray[np.float64]],
            executor: ThreadPoolExecutor | None,
        ) -> list[float]:
            # Reuse cached costs, then evaluate new candidates in bounded
            # batches while retaining their original poll order.
            nonlocal evaluations
            costs: list[float | None] = [None] * len(candidates)
            uncached_positions: list[int] = []
            uncached_candidates: list[npt.NDArray[np.float64]] = []
            for position, candidate in enumerate(candidates):
                key = tuple(np.round(candidate, 14))
                if key in cache:
                    costs[position] = cache[key]
                else:
                    uncached_positions.append(position)
                    uncached_candidates.append(candidate)
            uncached_costs: list[float] = []
            batch_size = self.max_batch_size or len(uncached_candidates) or 1
            for start in range(0, len(uncached_candidates), batch_size):
                batch = uncached_candidates[start:start + batch_size]
                if executor is None:
                    uncached_costs.extend(
                        evaluate_in_worker(candidate) for candidate in batch
                    )
                else:
                    uncached_costs.extend(executor.map(evaluate_in_worker, batch))
            for position, candidate, cost in zip(
                uncached_positions, uncached_candidates, uncached_costs,
                strict=True,
            ):
                key = tuple(np.round(candidate, 14))
                cache[key] = cost
                costs[position] = cost
                evaluations += 1
                emit_progress(progress_callback, ProgressEvent(
                    message=f"pattern-search evaluation {evaluations}: cost={cost:.8g}",
                    kind="evaluation",
                ))
            return [float(cost) for cost in costs if cost is not None]

        def improves(candidate_cost: float, reference_cost: float) -> bool:
            # Combine absolute and scale-relative tolerances so improvement
            # tests remain useful for both small and large objectives.
            tolerance = (
                self.objective_absolute_tolerance
                + self.objective_relative_tolerance
                * max(abs(candidate_cost), abs(reference_cost))
            )
            return candidate_cost < reference_cost - tolerance

        def direction_basis(iteration_index: int) -> tuple[str, npt.NDArray[np.float64]]:
            # Alternate direct coordinate moves with reproducible rotated moves
            # that can follow correlations between parameters.
            if iteration_index % 2 == 0:
                return "coordinate", np.eye(x.size, dtype=np.float64)
            samples = rng.standard_normal((x.size, x.size))
            basis, triangular = np.linalg.qr(samples)
            diagonal_sign = np.sign(np.diag(triangular))
            diagonal_sign[diagonal_sign == 0.0] = 1.0
            basis *= diagonal_sign
            return "orthonormal", basis

        def append_candidate(
            candidates: list[npt.NDArray[np.float64]],
            labels: list[str],
            seen: set[tuple[float, ...]],
            scheduled_uncached: set[tuple[float, ...]],
            candidate: npt.NDArray[np.float64],
            label: str,
        ) -> bool:
            # Clip to bounds and omit the incumbent, duplicate points, and new
            # evaluations that would exceed the objective budget.
            bounded = np.clip(candidate, 0.0, 1.0)
            if np.array_equal(bounded, x):
                return False
            key = tuple(np.round(bounded, 14))
            if key in seen:
                return False
            is_uncached = key not in cache
            if (
                is_uncached
                and evaluations + len(scheduled_uncached)
                >= self.max_evaluations
            ):
                return False
            seen.add(key)
            if is_uncached:
                scheduled_uncached.add(key)
            candidates.append(bounded)
            labels.append(label)
            return True

        # Evaluate the initial point before constructing the first poll mesh.
        started = time.perf_counter()
        rng = np.random.default_rng(self.random_seed)
        current_cost = evaluate(x)
        mesh_size = self.initial_mesh_size
        history: list[dict] = []
        iteration = 0
        status = "minimum_mesh_size"
        successful_displacement: npt.NDArray[np.float64] | None = None
        executor = (
            ThreadPoolExecutor(max_workers=self.parallel_workers)
            if self.parallel_workers > 1 else None
        )
        try:
            while (iteration < self.max_iterations and
                   evaluations < self.max_evaluations and
                   mesh_size >= self.minimum_mesh_size):
                best_x, best_cost = x, current_cost
                poll_candidates: list[npt.NDArray[np.float64]] = []
                candidate_labels: list[str] = []
                seen: set[tuple[float, ...]] = set()
                scheduled_uncached: set[tuple[float, ...]] = set()
                # Extrapolate along the previous successful displacement in
                # addition to polling the current direction basis.
                if (
                    successful_displacement is not None
                    and self.pattern_step_size > 0.0
                ):
                    append_candidate(
                        poll_candidates,
                        candidate_labels,
                        seen,
                        scheduled_uncached,
                        x + self.pattern_step_size * successful_displacement,
                        "pattern",
                    )

                basis_kind, basis = direction_basis(iteration)
                # A complete poll considers both signs of every basis vector;
                # the best improving candidate is chosen after all finish.
                for index in range(x.size):
                    direction = basis[:, index]
                    for sign in (-1.0, 1.0):
                        append_candidate(
                            poll_candidates,
                            candidate_labels,
                            seen,
                            scheduled_uncached,
                            x + sign * mesh_size * direction,
                            f"poll_{index}_{int(sign):+d}",
                        )

                if not poll_candidates:
                    status = "max_evaluations"
                    break
                costs = evaluate_poll(poll_candidates, executor)
                best_label: str | None = None
                for candidate, label, cost in zip(
                    poll_candidates, candidate_labels, costs, strict=True,
                ):
                    if improves(cost, best_cost):
                        best_x, best_cost, best_label = candidate, cost, label
                improved = improves(best_cost, current_cost)
                if improved:
                    # Accept the best point and optionally expand the mesh.
                    previous_x = x
                    x, current_cost = best_x, best_cost
                    successful_displacement = x - previous_x
                    mesh_size = min(
                        1.0,
                        mesh_size * self.mesh_expansion_factor,
                    )
                else:
                    # Only a failed complete poll contracts the mesh.
                    successful_displacement = None
                    mesh_size *= self.mesh_contraction_factor
                history.append({"iteration": iteration, "cost": current_cost,
                                "mesh_size": mesh_size, "evaluations": evaluations,
                                "improved": improved,
                                "basis": basis_kind,
                                "accepted_candidate": best_label,
                                "poll_size": len(poll_candidates)})
                iteration += 1
        finally:
            if executor is not None:
                executor.shutdown()

        if evaluations >= self.max_evaluations:
            status = "max_evaluations"
        elif iteration >= self.max_iterations:
            status = "max_iterations"

        # Re-evaluate the accepted endpoint so stateful objective diagnostics
        # describe it, rather than the last rejected point in the final poll.
        final_cost = checked_cost(evaluate_candidate(
            x, constitutive_law, parameter_map_size, state, metrics,
            objective_function, experiment_data,
        ))
        final_state = state.copy()
        final_state.update_from_normalised_degrees_of_freedom(x)
        runtime = time.perf_counter() - started
        return OptimisationOutcome(
            spatial_parameterisations=final_state.spatial_parameterisations,
            solve_result=SolveResult(
                solve_iteration=0, optimiser=snapshot_object(self),
                runtime_seconds=runtime, num_evaluations=evaluations + 1,
                success=status == "minimum_mesh_size", status=status,
                message=f"Generalized pattern search stopped at {status}.",
                initial_dofs=[float(dof.value) for dof in initial_dofs],
                final_dofs=[float(dof.value) for dof in final_state.collect_degrees_of_freedom()],
                final_objective={"cost": final_cost, "mesh_size": mesh_size,
                                 "iterations": iteration, "history": history,
                                 "components": _objective_components(objective_function),
                                 "active_lower_bounds": [bool(v <= 1e-8) for v in x],
                                 "active_upper_bounds": [bool(v >= 1-1e-8) for v in x]},
            ),
        )


def _objective_components(objective_function: IObjectiveFunction) -> dict:
    result = getattr(objective_function, "last_result", None)
    if result is None:
        return {}
    return asdict(result) if is_dataclass(result) else {"last_result": str(result)}


def _empty_outcome(spatial_parameterisations, optimiser) -> OptimisationOutcome:
    return OptimisationOutcome(
        spatial_parameterisations=spatial_parameterisations,
        solve_result=SolveResult(
            solve_iteration=0, optimiser=snapshot_object(optimiser),
            runtime_seconds=0.0, num_evaluations=0, success=True,
            status="skipped_no_dofs", message="No active degrees of freedom were available.",
            initial_dofs=[], final_dofs=[],
        ),
    )
