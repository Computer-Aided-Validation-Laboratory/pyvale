# ==============================================================================
# pyvale: the python validation engine
# License: MIT
# Copyright (C) 2025 The Computer Aided Validation Team
# ==============================================================================
from typing import Any
import itertools


def _single_sim_param_grid(in_params: dict[str,list[Any]] | None
                           ) -> list[dict[str,Any]] | None:
    """Private helper function that unpacks the list inside each dictionary key
    to find every possible unique combination of variables within each list.
    This gives a list of dictionaries where the dictionary values are single
    values not lists.

    Parameters
    ----------
    in_params : dict[str,list[Any]] | None
        Dictionary of lists keyed by the simulation variable name with lists
        containing the parameters the will form the grid of combinations.

    Returns
    -------
    list[dict[str,Any]] | None
        The list of unique dictionaries that have single values not lists.
    """
    if in_params is None:
        return None

    param_keys = in_params.keys()
    param_vals = in_params.values()
    param_combs = itertools.product(*param_vals)

    params = []
    for cc in param_combs:
        this_params = dict(zip(param_keys,cc))
        params.append(this_params)

    return params


def sweep_param_grid(in_params: list[dict[str,list[Any]] | None]
                          ) -> list[list[dict[str,Any] | None]]:
    """Helper function for generating grid parameter sweeps for all possible
    combinations of modified variables in the simulation chain.

    Parameters
    ----------
    in_params : list[dict[str,list[Any]] | None]
        List of the same length as the simulation input modifier/runner list
        where each dictionary corresponds to the variables that will be modified
        for the corresponding simulation tool. The variables should be keyed by
        their string identigfier in the simulation input file and the list
        should contain all unique values of the variables to analyse.

    Returns
    -------
    list[list[dict[str,Any] | None]]
        The outer list is for each unique parameter combination. The inner list
        position corresponds to the simulation input modifier/runner position in
        the simulation chain. The dictionary is then keyed by the variable name
        in the corresponding input file and the unique value the variable takes
        for this combination of parameters.
    """
    param_grids = []
    for pp in in_params:
        this_params = _single_sim_param_grid(pp)

        if this_params is None:
            param_grids.append([None])
        else:
            param_grids.append(this_params)

    sweep_params = [list(pp) for pp in itertools.product(*param_grids)]
    return sweep_params
