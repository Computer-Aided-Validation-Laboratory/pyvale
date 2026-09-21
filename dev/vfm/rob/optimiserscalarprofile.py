"""Experimental bounded cost profiling for one-parameter identification problems."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np
import numpy.typing as npt
from scipy.optimize import minimize_scalar


ScalarResidualFunction = Callable[[float], npt.ArrayLike]


@dataclass(slots=True, frozen=True)
class ScalarProfileResult:
    """Complete numerical evidence from a bounded scalar profile solve."""

    optimum: float
    minimum_cost: float
    residual: npt.NDArray[np.float64]
    classification: str
    scan_values: npt.NDArray[np.float64]
    scan_costs: npt.NDArray[np.float64]
    refined_values: npt.NDArray[np.float64]
    refined_costs: npt.NDArray[np.float64]
    near_minimum_lower: float
    near_minimum_upper: float
    near_minimum_cost_tolerance: float
    plateau_onset: float | None
    lower_bound_active: bool
    upper_bound_active: bool
    evaluation_count: int

    def to_summary(self) -> dict[str, object]:
        return {
            "optimum": self.optimum,
            "minimum_cost": self.minimum_cost,
            "classification": self.classification,
            "scan_values": self.scan_values.tolist(),
            "scan_costs": self.scan_costs.tolist(),
            "refined_values": self.refined_values.tolist(),
            "refined_costs": self.refined_costs.tolist(),
            "near_minimum_lower": self.near_minimum_lower,
            "near_minimum_upper": self.near_minimum_upper,
            "near_minimum_cost_tolerance": self.near_minimum_cost_tolerance,
            "plateau_onset": self.plateau_onset,
            "lower_bound_active": self.lower_bound_active,
            "upper_bound_active": self.upper_bound_active,
            "evaluation_count": self.evaluation_count,
            "residual_norm": float(np.linalg.norm(self.residual)),
            "residual_size": int(self.residual.size),
        }


def solve_scalar_profile(
    residual_function: ScalarResidualFunction,
    bounds: tuple[float, float],
    *,
    scan_points: int = 31,
    relative_cost_tolerance: float = 0.01,
    absolute_cost_tolerance: float = 1.0e-12,
    refinement_x_tolerance: float = 1.0e-5,
    weak_range_fraction: float = 0.25,
) -> ScalarProfileResult:
    """Scan a scalar interval, refine every distinct bracketed minimum, and classify.

    The objective is ``0.5 * residual @ residual``.  Endpoints are always retained
    as candidates.  This deliberately does not depend on an initial guess and is
    therefore safe when high yield values form an all-elastic zero-sensitivity
    plateau.
    """

    lower, upper = (float(value) for value in bounds)
    if not np.isfinite(lower) or not np.isfinite(upper) or lower >= upper:
        raise ValueError("bounds must be two finite increasing values")
    if scan_points < 5:
        raise ValueError("scan_points must be at least five")
    if relative_cost_tolerance < 0.0 or absolute_cost_tolerance < 0.0:
        raise ValueError("cost tolerances must be non-negative")

    evaluations = 0

    def evaluate(value: float) -> tuple[float, npt.NDArray[np.float64]]:
        nonlocal evaluations
        residual = np.asarray(residual_function(float(value)), dtype=np.float64).ravel()
        evaluations += 1
        if residual.size == 0 or not np.all(np.isfinite(residual)):
            return float("inf"), residual
        return float(0.5 * residual @ residual), residual

    scan_values = np.linspace(lower, upper, scan_points, dtype=np.float64)
    scan_costs = np.asarray([evaluate(value)[0] for value in scan_values])
    if not np.any(np.isfinite(scan_costs)):
        raise RuntimeError("Every scalar profile evaluation was non-finite")

    candidate_values = [lower, upper]
    candidate_costs = [float(scan_costs[0]), float(scan_costs[-1])]
    refined_values: list[float] = []
    refined_costs: list[float] = []
    for index in range(1, scan_points - 1):
        if scan_costs[index] <= scan_costs[index - 1] and scan_costs[index] <= scan_costs[index + 1]:
            result = minimize_scalar(
                lambda value: evaluate(float(value))[0],
                bounds=(float(scan_values[index - 1]), float(scan_values[index + 1])),
                method="bounded",
                options={"xatol": refinement_x_tolerance},
            )
            value = float(result.x)
            cost = float(result.fun)
            refined_values.append(value)
            refined_costs.append(cost)
            candidate_values.append(value)
            candidate_costs.append(cost)

    best_index = int(np.nanargmin(candidate_costs))
    optimum = float(candidate_values[best_index])
    minimum_cost, residual = evaluate(optimum)
    span = upper - lower
    boundary_tolerance = max(refinement_x_tolerance * 2.0, span * 1.0e-8)
    lower_active = optimum <= lower + boundary_tolerance
    upper_active = optimum >= upper - boundary_tolerance

    finite_costs = scan_costs[np.isfinite(scan_costs)]
    cost_scale = max(float(np.nanmax(finite_costs) - minimum_cost), minimum_cost, 1.0e-15)
    cost_tolerance = max(absolute_cost_tolerance, relative_cost_tolerance * cost_scale)
    near = scan_costs <= minimum_cost + cost_tolerance
    near_values = scan_values[near]
    if near_values.size:
        near_lower = float(near_values[0])
        near_upper = float(near_values[-1])
    else:
        near_lower = optimum
        near_upper = optimum

    plateau_onset = None
    tail_length = max(3, int(np.ceil(scan_points * 0.15)))
    high_tail_near = bool(np.all(near[-tail_length:]))
    if high_tail_near:
        first = scan_points - tail_length
        while first > 0 and near[first - 1]:
            first -= 1
        plateau_onset = float(scan_values[first])

    near_transitions = int(np.count_nonzero(np.diff(near.astype(np.int8)) != 0))
    disconnected_near_sets = near_transitions > 2
    near_width = near_upper - near_lower
    if high_tail_near:
        classification = "ONE_SIDED_OR_ELASTIC_PLATEAU"
    elif lower_active or upper_active:
        classification = "BOUND_LIMITED"
    elif disconnected_near_sets or near_width >= weak_range_fraction * span:
        classification = "WEAK_OR_MULTIMODAL"
    else:
        classification = "FINITE_CONDITIONAL_ESTIMATE"

    return ScalarProfileResult(
        optimum=optimum,
        minimum_cost=minimum_cost,
        residual=residual,
        classification=classification,
        scan_values=scan_values,
        scan_costs=scan_costs,
        refined_values=np.asarray(refined_values, dtype=np.float64),
        refined_costs=np.asarray(refined_costs, dtype=np.float64),
        near_minimum_lower=near_lower,
        near_minimum_upper=near_upper,
        near_minimum_cost_tolerance=cost_tolerance,
        plateau_onset=plateau_onset,
        lower_bound_active=lower_active,
        upper_bound_active=upper_active,
        evaluation_count=evaluations,
    )
