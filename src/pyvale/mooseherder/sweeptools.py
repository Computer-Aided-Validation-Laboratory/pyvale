# ==============================================================================
# pyvale: the python validation engine
# License: MIT
# Copyright (C) 2025 The Computer Aided Validation Team
# ==============================================================================

"""Construct Cartesian parameter grids keyed by simulation step name."""

import itertools

from .simrunner import InputValues


def sweep_param_grid(
    input_params: dict[str, dict[str, list[InputValues]]],
) -> list[dict[str, dict[str, InputValues]]]:
    """Construct a Cartesian product of candidate values for named simulation steps.

    Parameters
    ----------
    input_params : dict[str, dict[str, list[InputValues]]]
        Step names mapped to native parameter names and their candidate values.
        A vector-valued candidate is itself a list or tuple. An empty step mapping
        represents an unchanged step.

    Returns
    -------
    list[dict[str, dict[str, InputValues]]]
        All candidate combinations in dictionary insertion order, ready for
        MooseHerd.run. Each case owns its step/parameter dictionaries.

    Raises
    ------
    ValueError
        A parameter has an empty candidate list.
    """

    coordinates = []
    candidates = []
    for step, values in input_params.items():
        for name, options in values.items():
            if not options:
                raise ValueError(f"No candidate values for {step}/{name}.")
            coordinates.append((step, name))
            candidates.append(options)

    case_input_params = []
    for combination in itertools.product(*candidates):
        case: dict[str, dict[str, InputValues]] = {step: {} for step in input_params}
        for (step, name), value in zip(coordinates, combination):
            case[step][name] = value
        case_input_params.append(case)

    return case_input_params
