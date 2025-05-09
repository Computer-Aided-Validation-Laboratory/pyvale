# ================================================================================
# pyvale: the python validation engine
# License: MIT
# Copyright (C) 2025 The Computer Aided Validation Team
# ================================================================================

import matplotlib.pyplot as plt
import numpy as np

from pyvale.dic import dic2dcpp
from pyvale.dic.dic2d import DICdata_import
from pyvale.dic.dicresults import DICResults

def DICstrain(dic_data: DICResults | str,
                         window_size: int=5, 
                         window_element: int=4, 
                         strain_formulation: str="HENCKY",
                         binary: bool=False,
                         delimiter: str=" "):


    # Check the strain formulation is in the allowed list
    allowed_formulations = ["GREEN", "ALMANSI", "HENCKY", 
                            "BIOT_EULER", "BIOT_LAGRANGE"]
    
    if strain_formulation not in allowed_formulations:
        raise ValueError(f"Invalid strain formulation: "
                         f"'{strain_formulation}'. Allowed values are: "
                         f"{', '.join(allowed_formulations)}.")

    # check the strain window element is one of the allowed values
    allowed_element = [4, 9]
    if window_element not in allowed_element:
        raise ValueError(f"Invalid strain window element type: "
                         f"Q{window_element}. Allowed values are: "
                         f"{', '.join(map(str, allowed_element))}.")

    # chceck the window size is an odd number
    if window_size % 2 == 0:
        raise ValueError(f"Invalid strain window size: '{window_size}'. "
                         f"Must be an odd number.")


    if type(dic_data) is str:
        results = DICdata_import(layout="matrix", data=dic_data,
                                 binary=binary, delimiter=delimiter)
    else:
        results = dic_data

    nss_x = results.ss_x.shape[1]
    nss_y = results.ss_y.shape[0]
    nimg = results.u.shape[0]

    dic2dcpp.strain_engine(results.ss_x,results.ss_y,
                           results.u,results.v,
                           nss_x, nss_y, nimg,
                           window_size, window_element, strain_formulation)



