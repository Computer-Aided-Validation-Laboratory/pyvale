"""Construct Cartesian parameter grids keyed by simulation step name."""

import itertools

from .simrunner import MooseValue


def sweep_param_grid(
    parameters: dict[str, dict[str, list[MooseValue]]],
) -> list[dict[str, dict[str, MooseValue]]]:
    """Return all combinations in insertion order, including unchanged steps.

    Each outer key is a SimulationStep name. Each inner key is a parameter name
    whose list gives candidate values. A vector-valued candidate is itself a
    list or tuple. Empty candidate lists are rejected.
    """
    coordinates = []
    candidates = []
    for step, values in parameters.items():
        for name, options in values.items():
            if not options:
                raise ValueError(f"No candidate values for {step}/{name}.")
            coordinates.append((step, name))
            candidates.append(options)
    cases = []
    for combination in itertools.product(*candidates):
        case: dict[str, dict[str, MooseValue]] = {step: {} for step in parameters}
        for (step, name), value in zip(coordinates, combination):
            case[step][name] = value
        cases.append(case)
    return cases
