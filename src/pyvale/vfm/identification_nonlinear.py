from __future__ import annotations

from copy import deepcopy

import numpy as np
import numpy.typing as npt
from pymoo.algorithms.soo.nonconvex.pattern import PatternSearch
from pymoo.core.problem import Problem
from pymoo.optimize import minimize
from scipy.optimize import least_squares

from pyvale.vfm.mechanical_properties import (
    KnownParameter,
    MechanicalProperties,
    coerce_parameter_name,
)
from pyvale.vfm.metrics import MetricContext, build_metric, evaluate_metrics
from pyvale.vfm.project_definition import PhaseDefinition, PhaseResult, TestData
from pyvale.vfm.radial_return import radial_return
from pyvale.vfm.spatial_parameterisation import (
    ParameterState,
    collect_active_dofs,
    pack_dof_vector,
    resolve_parameter_maps,
    update_parameter_states_from_vector,
)


def run_nonlinear_identification(
    test_data: TestData,
    phase_definition: PhaseDefinition,
    base_mechanical_properties: MechanicalProperties,
    parameter_states: dict[str, ParameterState],
) -> PhaseResult:
    """Run one nonlinear identification phase with explicit objective steps."""

    print(f"  Preparing nonlinear identification for {phase_definition.name}.")
    for parameter_state in parameter_states.values():
        parameter_state.prepare(test_data)

    active_dofs = collect_active_dofs(parameter_states)
    print(f"  Active DOFs: {len(active_dofs)}")
    metric_context = MetricContext(
        phase_definition=phase_definition,
        base_mechanical_properties=base_mechanical_properties,
        parameter_states=parameter_states,
        active_dofs=active_dofs,
    )

    metrics_with_weights = []
    for metric_spec in phase_definition.metrics:
        metric = build_metric(metric_spec)
        metric.prepare(test_data, metric_context)
        metrics_with_weights.append((metric, metric_spec.weight))
        print(
            f"  Prepared metric '{metric_spec.kind}' "
            f"with weight {metric_spec.weight:.6g}."
        )

    if not metrics_with_weights:
        raise ValueError(
            f"Phase '{phase_definition.name}' does not define any metrics."
        )

    initial_vector, bounds = pack_dof_vector(active_dofs)
    print(
        f"  Initial optimiser vector has shape {initial_vector.shape}."
    )

    best = {
        "cost": np.inf,
        "vector": initial_vector.copy(),
        "metric_values": {},
        "parameter_maps": {},
        "stress": None,
        "equivalent_stress": None,
        "yield_map": None,
        "equivalent_plastic_strain": None,
        "parameter_states": deepcopy(parameter_states),
    }
    evaluation_counter = {"count": 0}

    def evaluate_candidate(vector: npt.NDArray[np.float64]) -> float:
        evaluation_counter["count"] += 1
        update_parameter_states_from_vector(parameter_states, active_dofs, vector)
        parameter_maps = resolve_parameter_maps(parameter_states, test_data)
        resolved_properties = _resolve_mechanical_properties(
            base_mechanical_properties,
            parameter_maps,
        )

        stress, equivalent_stress, yield_map, equivalent_plastic_strain = radial_return(
            test_data.strain,
            resolved_properties,
        )

        cost, metric_values, _ = evaluate_metrics(
            metrics_with_weights,
            stress,
            test_data,
            MetricContext(
                phase_definition=phase_definition,
                base_mechanical_properties=base_mechanical_properties,
                resolved_mechanical_properties=resolved_properties,
                parameter_states=parameter_states,
                active_dofs=active_dofs,
                parameter_maps=parameter_maps,
            ),
        )

        if cost < best["cost"]:
            best["cost"] = float(cost)
            best["vector"] = vector.copy()
            best["metric_values"] = metric_values
            best["parameter_maps"] = {
                name: parameter_map.copy()
                for name, parameter_map in parameter_maps.items()
            }
            best["stress"] = stress.copy()
            best["equivalent_stress"] = equivalent_stress.copy()
            best["yield_map"] = yield_map.copy()
            best["equivalent_plastic_strain"] = equivalent_plastic_strain.copy()
            best["parameter_states"] = deepcopy(parameter_states)
            print(
                f"    Eval {evaluation_counter['count']}: "
                f"new best cost = {float(cost):.6g}"
            )
        elif evaluation_counter["count"] % 25 == 0:
            print(
                f"    Eval {evaluation_counter['count']}: "
                f"current cost = {float(cost):.6g}"
            )

        return float(cost)

    if active_dofs:
        print(f"  Starting optimiser '{phase_definition.optimiser.kind}'.")
        best_vector = _run_optimiser(
            phase_definition=phase_definition,
            objective_function=evaluate_candidate,
            initial_vector=initial_vector,
            bounds=bounds,
        )
        evaluate_candidate(best_vector)
    else:
        print("  No active DOFs, evaluating the fixed candidate once.")
        evaluate_candidate(initial_vector)

    print(
        f"  Finished phase {phase_definition.name} after "
        f"{evaluation_counter['count']} evaluation(s). Best cost = "
        f"{float(best['cost']):.6g}."
    )

    return PhaseResult(
        phase_name=phase_definition.name,
        cost=float(best["cost"]),
        metric_values=best["metric_values"],
        parameter_maps=best["parameter_maps"],
        stress=best["stress"],
        equivalent_stress=best["equivalent_stress"],
        yield_map=best["yield_map"],
        equivalent_plastic_strain=best["equivalent_plastic_strain"],
        best_dof_vector=best["vector"],
        parameter_states=best["parameter_states"],
    )


def _resolve_mechanical_properties(
    base_mechanical_properties: MechanicalProperties,
    parameter_maps: dict[str, npt.NDArray[np.float64]],
) -> MechanicalProperties:
    resolved_parameters = dict(base_mechanical_properties.parameters)

    for parameter_name, parameter_map in parameter_maps.items():
        resolved_parameters[coerce_parameter_name(parameter_name)] = KnownParameter(
            parameter_map
        )

    return MechanicalProperties(
        constituitive_law=base_mechanical_properties.constituitive_law,
        parameters=resolved_parameters,
    )


def _run_optimiser(
    phase_definition: PhaseDefinition,
    objective_function,
    initial_vector: npt.NDArray[np.float64],
    bounds: list[tuple[float, float]],
) -> npt.NDArray[np.float64]:
    optimiser_kind = phase_definition.optimiser.kind

    if optimiser_kind == "least_squares":
        return _run_least_squares(
            objective_function,
            initial_vector,
            bounds,
            phase_definition.optimiser.options,
        )

    if optimiser_kind == "pattern_search":
        return _run_pattern_search(
            objective_function,
            initial_vector,
            bounds,
            phase_definition.optimiser.options,
        )

    raise ValueError(f"Unsupported optimiser '{optimiser_kind}'.")


def _run_least_squares(
    objective_function,
    initial_vector: npt.NDArray[np.float64],
    bounds: list[tuple[float, float]],
    options: dict[str, object],
) -> npt.NDArray[np.float64]:
    lower_bounds = np.array([lower for lower, _ in bounds], dtype=np.float64)
    upper_bounds = np.array([upper for _, upper in bounds], dtype=np.float64)

    method = str(options.get("method", "lm"))
    if method == "lm" and initial_vector.size > 1:
        # `lm` needs at least as many residuals as variables. The current
        # scientific toolkit uses a scalar aggregated cost, so we fall back
        # to `trf` for multi-DOF problems.
        method = "trf"
        print("    Switched least_squares method from 'lm' to 'trf' for multi-DOF scalar cost.")

    verbose = int(options.get("verbose", 2))
    print(
        f"    least_squares(method={method}, max_nfev={int(options.get('max_nfev', 200))}, "
        f"verbose={verbose})"
    )

    result = least_squares(
        lambda vector: np.array([objective_function(vector)], dtype=np.float64),
        x0=initial_vector,
        bounds=(lower_bounds, upper_bounds),
        method=method,
        max_nfev=int(options.get("max_nfev", 200)),
        ftol=float(options.get("ftol", 1e-8)),
        xtol=float(options.get("xtol", 1e-8)),
        gtol=float(options.get("gtol", 1e-8)),
        verbose=verbose,
    )

    print(
        f"    least_squares finished with status {result.status}: "
        f"{result.message.strip()}"
    )

    return np.asarray(result.x, dtype=np.float64)


def _run_pattern_search(
    objective_function,
    initial_vector: npt.NDArray[np.float64],
    bounds: list[tuple[float, float]],
    options: dict[str, object],
) -> npt.NDArray[np.float64]:
    lower_bounds = np.array([lower for lower, _ in bounds], dtype=np.float64)
    upper_bounds = np.array([upper for _, upper in bounds], dtype=np.float64)
    verbose = bool(options.get("verbose", True))
    print(
        f"    pattern_search(max_evaluations={int(options.get('max_evaluations', 200))}, "
        f"seed={int(options.get('seed', 1))}, verbose={verbose})"
    )

    class ScalarObjectiveProblem(Problem):
        def __init__(self) -> None:
            super().__init__(
                n_var=initial_vector.size,
                n_obj=1,
                n_ieq_constr=0,
                xl=lower_bounds,
                xu=upper_bounds,
            )

        def _evaluate(self, x, out, *args, **kwargs):
            candidates = np.atleast_2d(x)
            values = np.array(
                [objective_function(candidate) for candidate in candidates],
                dtype=np.float64,
            )
            out["F"] = values[:, np.newaxis]

    algorithm = PatternSearch()
    result = minimize(
        ScalarObjectiveProblem(),
        algorithm,
        x0=initial_vector,
        verbose=verbose,
        seed=int(options.get("seed", 1)),
        termination=("n_eval", int(options.get("max_evaluations", 200))),
    )

    print("    pattern_search finished.")

    return np.asarray(result.X, dtype=np.float64)
