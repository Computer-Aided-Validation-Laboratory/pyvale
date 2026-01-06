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
                      exp_data: dict[tuple[str,...],np.ndarray]) -> None:
    """Saves the results of a simulated experiment to disk.

    Parameters
    ----------
    save_file : Path
        Path including file name to where the simulated experiment data should 
        be saved.
    exp_data : dict[tuple[str,...],np.ndarray]
        The simulated experiment data dictionary to save.
    """

    flat_keys = []
    key_lens = []        
    save_dict = {}
    for kk,vv in exp_data.items():
        key_lens.append(len(kk))
        for ss in kk:
            flat_keys.append(ss)
    
        new_key = "-".join(kk)
        save_dict[new_key] = vv
        
    save_dict["keys"] = np.array(flat_keys,dtype='U')
    save_dict["key_lens"] = np.array(key_lens,dtype=np.uint8) 
    
    # exp_data:
    # dict[tuple[str,...],shape=(n_exps,n_sens,n_comps,n_time_steps)]
    # The ** operator unpacks the dictionary into function keyword arguments
    np.savez(save_file,**save_dict,allow_pickle=False)


def load_exp_sim_data(load_file: Path) -> dict[str,np.ndarray]:
    """Loads the results of a simulated experiment from disk.

    Parameters
    ----------
    load_file : Path
        Path and file name for the file where the data should be loaded from. 

    Returns
    -------
    dict[tuple[str,...],np.ndarray]
        The simulated experiment data dictionary loaded from disk.
    """
    # NOTE: npz files are loaded in a 'lazy' manner so we must use a context
    # manager here and convert to a dictionary which forces everything in the
    # npz to be directly loaded into memory.
    flat_data = {}
    with np.load(load_file) as npzfile:
        flat_data = dict(npzfile)
    
    flat_keys = flat_data["keys"]
    key_lens = flat_data["key_lens"]
    flat_data.pop("keys")
    flat_data.pop("key_lens")
    
    exp_data = {}
    key_ind: int = 0
    for key_len in key_lens:
        tuple_key = tuple(flat_keys[key_ind:key_ind+key_len])
        key_ind += key_len

        for join_key in flat_data:
            str_key_count: int = 0
            
            for str_key in tuple_key:
                if str_key in join_key:
                    str_key_count += 1

            if str_key_count == key_len:
                exp_data[tuple_key] = flat_data[join_key]
                flat_data.pop(join_key)
                break
            
    # dict[tuple[str,...],shape=(n_sims,n_exps,n_sens,n_comps,n_time_steps)]
    return exp_data

