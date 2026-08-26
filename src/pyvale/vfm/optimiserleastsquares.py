import time

import numpy as np
import numpy.typing as npt
from scipy.optimize import least_squares

from pyvale.vfm.constlaw import IConstitutiveLaw
from pyvale.vfm.experimentdata import ExperimentData
from pyvale.vfm.identificationresult import (
    OptimisationOutcome,
    SolveResult,
    snapshot_object,
)
from pyvale.vfm.metric import IMetric
from pyvale.vfm.objectivefunc import (
    IObjectiveFunction,
    IVectorObjectiveFunction,
)
from pyvale.vfm.optimiser import (
    IOptimiser,
    evaluate_candidate,
)
from pyvale.vfm.spatialparam import (
    PhaseSpatialState,
    ISpatialParameterisation,
)


class OptimiserLeastSquares(IOptimiser):
    """
    Least-squares optimiser driving the parameter search.

    Wraps ``scipy.optimize.least_squares`` to minimise a vector objective
    over active, normalised degrees of freedom. Bounds of ``[0, 1]`` keep
    every degree of freedom within its configured physical bounds.
    """

    def __init__(self, *, max_evaluations: int | None = None) -> None:
        if max_evaluations is not None and max_evaluations < 1:
            raise ValueError("max_evaluations must be positive or None.")
        self.max_evaluations = max_evaluations

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
        _ = progress_callback
        phase_spatial_state = PhaseSpatialState(spatial_parameterisations)
        dofs = phase_spatial_state.collect_normalised_degrees_of_freedom()
        initial_dofs = [
            float(dof.value)
            for dof in phase_spatial_state.collect_degrees_of_freedom()
        ]
        if dofs.size == 0:
            return OptimisationOutcome(
                spatial_parameterisations=spatial_parameterisations,
                solve_result=SolveResult(
                    solve_iteration=0,
                    optimiser=snapshot_object(
                        self,
                        options={
                            "method": "trf",
                            "max_evaluations": self.max_evaluations,
                        },
                    ),
                    runtime_seconds=0.0,
                    num_evaluations=0,
                    success=True,
                    status="skipped_no_dofs",
                    message="No active degrees of freedom were available.",
                    initial_dofs=[],
                    final_dofs=[],
                ),
            )

        started_at = time.perf_counter()
        result = least_squares(
            evaluate_candidate,
            dofs,
            bounds=(np.zeros_like(dofs), np.ones_like(dofs)),
            method="trf",
            max_nfev=self.max_evaluations,
            args=(
                constitutive_law,
                parameter_map_size,
                phase_spatial_state,
                metrics,
                objective_function,
                experiment_data,
            )
        )
        runtime_seconds = time.perf_counter() - started_at

        optimised_phase_spatial_state = phase_spatial_state.copy()
        optimised_phase_spatial_state.update_from_normalised_degrees_of_freedom(
            result.x
        )

        return OptimisationOutcome(
            spatial_parameterisations=optimised_phase_spatial_state.spatial_parameterisations,
            solve_result=SolveResult(
                solve_iteration=0,
                optimiser=snapshot_object(
                    self,
                    options={
                        "method": "trf",
                        "max_evaluations": self.max_evaluations,
                    },
                ),
                runtime_seconds=runtime_seconds,
                num_evaluations=int(result.nfev),
                success=bool(result.success),
                status=int(result.status),
                message=str(result.message),
                initial_dofs=initial_dofs,
                final_dofs=[
                    float(dof.value)
                    for dof in optimised_phase_spatial_state.collect_degrees_of_freedom()
                ],
                final_objective=_summarise_least_squares_result(result),
            ),
        )


def _summarise_least_squares_result(
    result,
) -> dict:
    residual = np.asarray(result.fun, dtype=np.float64).ravel()
    finite_residual = residual[np.isfinite(residual)]
    residual_norm = (
        None
        if finite_residual.size == 0
        else float(np.linalg.norm(finite_residual))
    )
    return {
        "cost": float(result.cost),
        "residual_norm": residual_norm,
        "residual_size": int(residual.size),
        "finite_residual_count": int(finite_residual.size),
        "optimality": float(result.optimality),
    }
