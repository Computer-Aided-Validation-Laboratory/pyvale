# ==============================================================================
# pyvale: the python validation engine
# License: MIT
# Copyright (C) 2025 The Computer Aided Validation Team
# ==============================================================================

"""
This module contains functions for saving/loading the results of simulated
experiments with virtual sensor arrays.
"""

from pathlib import Path
import numpy as np


def save_exp_sim_data(save_file: Path,
                      exp_data: dict[str,np.ndarray]) -> None:
    """Saves the results of a simulated experiment to disk.

    Parameters
    ----------
    save_file : Path
        Path including file name to where the simulated experiment data should 
        be saved.
    exp_data : dict[str,np.ndarray]
        The simulated experiment data dictionary to save.
    """
    # exp_data:
    # dict[str,shape=(n_sims,n_exps,n_sens,n_comps,n_time_steps)]
    # The ** operator unpacks the dictionary into function keyword arguments
    np.savez(save_file,**exp_data,allow_pickle=False)


def load_exp_sim_data(load_file: Path) -> dict[str,np.ndarray]:
    """Loads the results of a simulated experiment from disk.

    Parameters
    ----------
    load_file : Path
        Path and file name for the file where the data should be loaded from. 

    Returns
    -------
    dict[str,np.ndarray]
        The simulated experiment data dictionary loaded from disk.
    """
    # NOTE: npz files are loaded in a 'lazy' manner so we must use a context
    # manager here and convert to a dictionary which forces everything in the
    # npz to be directly loaded into memory.
    exp_data = {}
    with np.load(load_file) as npzfile:
        exp_data = dict(npzfile)

    # dict[str,shape=(n_sims,n_exps,n_sens,n_comps,n_time_steps)]
    return exp_data

