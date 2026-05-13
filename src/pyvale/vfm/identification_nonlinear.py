from __future__ import annotations

from copy import deepcopy

import numpy as np
import numpy.typing as npt
from pymoo.algorithms.soo.nonconvex.pattern import PatternSearch
from pymoo.core.problem import Problem
from pymoo.optimize import minimize
from scipy.optimize import least_squares

from pyvale.vfm.mechanical_properties import (
    EParameterName,
    KnownParameter,
    MechanicalProperties,
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
        # Intantiate MetricSpec object from user-defined spec
        metric = build_metric(metric_spec)
        # Prepare reusable metric data before optimisation (e.g. sbvf mesh)
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

    if phase_definition.optimiser.kind == "independent_slices":
        return _run_independent_slices_identification(
            test_data=test_data,
            phase_definition=phase_definition,
            base_mechanical_properties=base_mechanical_properties,
            parameter_states=parameter_states,
            active_dofs=active_dofs,
            metrics_with_weights=metrics_with_weights,
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

    def evaluate_candidate(
        vector: npt.NDArray[np.float64],
    ) -> tuple[float, npt.NDArray[np.float64]]:
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

        cost, metric_values, metric_results = evaluate_metrics(
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
        residual_vector = _build_least_squares_residual_vector(
            metrics_with_weights,
            metric_results,
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

        return float(cost), residual_vector

    def evaluate_candidate_cost(vector: npt.NDArray[np.float64]) -> float:
        cost, _ = evaluate_candidate(vector)
        return cost

    def evaluate_candidate_residuals(vector: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        _, residual_vector = evaluate_candidate(vector)
        return residual_vector

    if active_dofs:
        print(f"  Starting optimiser '{phase_definition.optimiser.kind}'.")
        best_vector = _run_optimiser(
            phase_definition=phase_definition,
            objective_function=evaluate_candidate_cost,
            residual_function=evaluate_candidate_residuals,
            initial_vector=initial_vector,
            bounds=bounds,
        )
        evaluate_candidate_cost(best_vector)
    else:
        print("  No active DOFs, evaluating the fixed candidate once.")
        evaluate_candidate_cost(initial_vector)

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
        resolved_parameters[EParameterName[parameter_name]] = KnownParameter(
            parameter_map
        )

    return MechanicalProperties(
        constituitive_law=base_mechanical_properties.constituitive_law,
        parameters=resolved_parameters,
    )


def _run_independent_slices_identification(
    test_data: TestData,
    phase_definition: PhaseDefinition,
    base_mechanical_properties: MechanicalProperties,
    parameter_states: dict[str, ParameterState],
    active_dofs,
    metrics_with_weights,
) -> PhaseResult:
    from pyvale.vfm.metric_udvf_slicewise import UDVFSlicewiseMetric
    from pyvale.vfm.parameterisation_slice import (
        SlicePartition,
        SliceWiseParameterisation,
        slice_partitions_match,
    )

    if not active_dofs:
        raise ValueError(
            "The 'independent_slices' optimiser requires active slicewise DOFs."
        )

    slicewise_metrics = [
        (metric, weight)
        for metric, weight in metrics_with_weights
        if isinstance(metric, UDVFSlicewiseMetric)
    ]
    if len(metrics_with_weights) != 1 or len(slicewise_metrics) != 1:
        raise ValueError(
            "The 'independent_slices' optimiser currently supports exactly one "
            "'udvf_slicewise' metric."
        )

    slicewise_metric, metric_weight = slicewise_metrics[0]
    slicewise_parameterisations = _collect_slicewise_parameterisations(parameter_states)
    common_partition = _extract_common_slice_partition(slicewise_parameterisations)

    if len(active_dofs) != len(slicewise_parameterisations) * common_partition.num_slices:
        raise ValueError(
            "The 'independent_slices' optimiser currently requires every active DOF "
            "to belong to a slicewise parameterisation."
        )

    local_solver_kind = str(
        phase_definition.optimiser.options.get("local_solver", "least_squares")
    )
    print(
        f"  Starting optimiser 'independent_slices' with "
        f"{common_partition.num_slices} slice(s) and local solver "
        f"'{local_solver_kind}'."
    )

    total_local_evaluations = 0

    for slice_index, slice_subdomain in enumerate(common_partition.slice_subdomains):
        local_dofs = [
            parameterisation.slice_dof(slice_index)
            for parameterisation in slicewise_parameterisations
            if parameterisation.slice_dof(slice_index).active
        ]
        initial_vector, bounds = pack_dof_vector(local_dofs)
        print(
            f"    Slice {slice_index + 1}/{common_partition.num_slices}: "
            f"{len(local_dofs)} local DOF(s)."
        )

        local_evaluation_counter = {"count": 0}
        local_best = {
            "cost": np.inf,
            "vector": initial_vector.copy(),
        }

        def evaluate_local_candidate(
            vector: npt.NDArray[np.float64],
        ) -> tuple[float, npt.NDArray[np.float64]]:
            local_evaluation_counter["count"] += 1
            total_vector_update = {
                dof.uid: float(value)
                for dof, value in zip(local_dofs, vector, strict=True)
            }
            for parameter_state in parameter_states.values():
                for parameterisation in parameter_state.parameterisations:
                    parameterisation.update_from_values(total_vector_update)

            parameter_maps = resolve_parameter_maps(parameter_states, test_data)
            local_parameter_maps = {
                parameter_name: parameter_map[
                    slice_subdomain.row_slice,
                    slice_subdomain.col_slice,
                ].copy()
                for parameter_name, parameter_map in parameter_maps.items()
            }
            resolved_properties = _resolve_mechanical_properties(
                base_mechanical_properties,
                local_parameter_maps,
            )
            stress, _, _, _ = radial_return(
                slice_subdomain.local_test_data.strain,
                resolved_properties,
            )
            metric_result = slicewise_metric.evaluate_single_slice(
                stress=stress,
                test_data=slice_subdomain.local_test_data,
                slice_width=float(common_partition.slice_widths[slice_index]),
                slice_index=slice_index,
            )
            raw_residual = np.asarray(
                metric_result.details["residual_vector"],
                dtype=np.float64,
            ).reshape(-1)
            residual_vector = np.sqrt(metric_weight) * raw_residual
            cost = float(residual_vector @ residual_vector)

            if cost < local_best["cost"]:
                local_best["cost"] = cost
                local_best["vector"] = vector.copy()

            return cost, residual_vector

        def evaluate_local_cost(vector: npt.NDArray[np.float64]) -> float:
            cost, _ = evaluate_local_candidate(vector)
            return cost

        def evaluate_local_residual(vector: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
            _, residual_vector = evaluate_local_candidate(vector)
            return residual_vector

        if local_dofs:
            best_local_vector = _run_local_slice_optimiser(
                solver_kind=local_solver_kind,
                options=phase_definition.optimiser.options,
                objective_function=evaluate_local_cost,
                residual_function=evaluate_local_residual,
                initial_vector=initial_vector,
                bounds=bounds,
            )
            evaluate_local_cost(best_local_vector)

        total_local_evaluations += local_evaluation_counter["count"]
        print(
            f"      Completed slice {slice_index + 1} after "
            f"{local_evaluation_counter['count']} evaluation(s). "
            f"Best local cost = {float(local_best['cost']):.6g}."
        )

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
    best_vector, _ = pack_dof_vector(active_dofs)

    print(
        f"  Finished phase {phase_definition.name} after "
        f"{total_local_evaluations} local evaluation(s). Best cost = "
        f"{float(cost):.6g}."
    )

    return PhaseResult(
        phase_name=phase_definition.name,
        cost=float(cost),
        metric_values=metric_values,
        parameter_maps=parameter_maps,
        stress=stress,
        equivalent_stress=equivalent_stress,
        yield_map=yield_map,
        equivalent_plastic_strain=equivalent_plastic_strain,
        best_dof_vector=best_vector,
        parameter_states=deepcopy(parameter_states),
    )


def _collect_slicewise_parameterisations(
    parameter_states: dict[str, ParameterState],
):
    from pyvale.vfm.parameterisation_slice import SliceWiseParameterisation

    slicewise_parameterisations: list[SliceWiseParameterisation] = []
    non_slicewise_active_dofs: list[str] = []

    for parameter_state in parameter_states.values():
        for parameterisation in parameter_state.parameterisations:
            active_dofs = parameterisation.active_dofs()
            if not active_dofs:
                continue

            if isinstance(parameterisation, SliceWiseParameterisation):
                slicewise_parameterisations.append(parameterisation)
            else:
                non_slicewise_active_dofs.extend(dof.uid for dof in active_dofs)

    if non_slicewise_active_dofs:
        joined = ", ".join(non_slicewise_active_dofs)
        raise ValueError(
            "The 'independent_slices' optimiser only supports slicewise active "
            f"DOFs. Found: {joined}."
        )

    return slicewise_parameterisations


def _extract_common_slice_partition(slicewise_parameterisations):
    from pyvale.vfm.parameterisation_slice import slice_partitions_match

    if not slicewise_parameterisations:
        raise ValueError(
            "The 'independent_slices' optimiser needs at least one slicewise "
            "parameterisation."
        )

    common_partition = slicewise_parameterisations[0].partition
    if common_partition is None:
        raise ValueError(
            "Slicewise parameterisations must be prepared before optimisation."
        )

    for parameterisation in slicewise_parameterisations[1:]:
        if parameterisation.partition is None:
            raise ValueError(
                "Slicewise parameterisations must be prepared before optimisation."
            )
        if not slice_partitions_match(common_partition, parameterisation.partition):
            raise ValueError(
                "All slicewise parameterisations in an 'independent_slices' phase "
                "must share the same slice layout."
            )

    return common_partition


def _run_local_slice_optimiser(
    solver_kind: str,
    options: dict[str, object],
    objective_function,
    residual_function,
    initial_vector: npt.NDArray[np.float64],
    bounds: list[tuple[float, float]],
) -> npt.NDArray[np.float64]:
    if solver_kind == "least_squares":
        return _run_least_squares(
            residual_function=residual_function,
            initial_vector=initial_vector,
            bounds=bounds,
            options=options,
        )

    if solver_kind == "pattern_search":
        return _run_pattern_search(
            objective_function=objective_function,
            initial_vector=initial_vector,
            bounds=bounds,
            options=options,
        )

    raise ValueError(
        "The 'independent_slices' optimiser currently supports local_solver "
        "'least_squares' or 'pattern_search'."
    )


def _run_optimiser(
    phase_definition: PhaseDefinition,
    objective_function,
    residual_function,
    initial_vector: npt.NDArray[np.float64],
    bounds: list[tuple[float, float]],
) -> npt.NDArray[np.float64]:
    optimiser_kind = phase_definition.optimiser.kind

    if optimiser_kind == "least_squares":
        return _run_least_squares(
            residual_function,
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
    residual_function,
    initial_vector: npt.NDArray[np.float64],
    bounds: list[tuple[float, float]],
    options: dict[str, object],
) -> npt.NDArray[np.float64]:
    lower_bounds = np.array([lower for lower, _ in bounds], dtype=np.float64)
    upper_bounds = np.array([upper for _, upper in bounds], dtype=np.float64)
    initial_residual = np.asarray(residual_function(initial_vector), dtype=np.float64)

    method = str(options.get("method", "lm"))
    if method == "lm" and np.any(np.isfinite(lower_bounds) | np.isfinite(upper_bounds)):
        method = "trf"
        print("    Switched least_squares method from 'lm' to 'trf' because bounded problems are not supported by 'lm'.")
    if method == "lm" and initial_residual.size < initial_vector.size:
        method = "trf"
        print("    Switched least_squares method from 'lm' to 'trf' because the residual vector is shorter than the parameter vector.")

    verbose = int(options.get("verbose", 2))
    print(
        f"    least_squares(method={method}, max_nfev={int(options.get('max_nfev', 200))}, "
        f"verbose={verbose})"
    )

    result = least_squares(
        residual_function,
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


def _build_least_squares_residual_vector(
    metrics_with_weights,
    metric_results,
) -> npt.NDArray[np.float64]:
    residual_chunks: list[npt.NDArray[np.float64]] = []

    for (_, weight), result in zip(metrics_with_weights, metric_results, strict=True):
        raw_residual = result.details.get("residual_vector")
        if raw_residual is None:
            residual_chunks.append(
                np.array([np.sqrt(max(0.0, weight * result.value))], dtype=np.float64)
            )
            continue

        residual_vector = np.asarray(raw_residual, dtype=np.float64).reshape(-1)
        if residual_vector.size == 0:
            residual_chunks.append(np.zeros(0, dtype=np.float64))
            continue

        residual_chunks.append(np.sqrt(weight) * residual_vector)

    if not residual_chunks:
        return np.zeros(0, dtype=np.float64)

    return np.concatenate(residual_chunks)


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
