# ================================================================================
# pyvale: the python validation engine
# License: MIT
# Copyright (C) 2025 The Computer Aided Validation Team
# ================================================================================

import matplotlib.pyplot as plt
import numpy as np

from pyvale import dic2dcpp
from pyvale.dicdataimport import dic_data_import
from pyvale.dicresults import DICResults

def strain_2d(data: DICResults | str,
                         window_size: int=5, 
                         window_element: int=4,
                         input_binary: bool=False,
                         input_delimiter: str=" ",
                         output_def_grad: bool=True,
                         output_strain: bool=True,
                         output_basepath: str="./",
                         output_binary: bool=False,
                         output_prefix: str="strain_",
                         output_delimiter: str=" ",
                         output_at_end: bool=False,
                         strain_formulation: str="HENCKY"):
    """
    Compute strain fields from DIC displacement data using a finite element smoothing approach.

    This function validates the input data and parameters, optionally loads DIC results from file,
    and passes the data to a C++-accelerated backend for strain computation.

    Parameters
    ----------
    data : DICResults or str
        A `DICResults` instance containing displacement and subset coordinates,
        OR a path to files from which the data should be imported.
    window_size : int, optional
        The size of the local window over which to compute strain (must be odd), by default 5.
    window_element : int, optional
        The type of finite element shape function used in the strain window: 4 (bilinear) or 9 (biquadratic),
        by default 4.
    strain_formulation : str, optional
        The strain definition to use: one of 'GREEN', 'ALMANSI', 'HENCKY', 'BIOT_EULER', 'BIOT_LAGRANGE'.
        Defaults to 'HENCKY'.
    binary : bool, optional
        Whether the input file is in binary format. Only relevant if `data` is a file path.
    delimiter : str, optional
        The delimiter used in the input file if it's in text format, by default " ".

    Raises
    ------
    ValueError
        If any of the input parameters are invalid (e.g., unsupported strain formulation,
        even window size, or invalid element type).
    """

    allowed_formulations = ["GREEN", "ALMANSI", "HENCKY", "BIOT_EULER", "BIOT_LAGRANGE"]
    if strain_formulation not in allowed_formulations:
        raise ValueError(f"Invalid strain formulation: '{strain_formulation}'. "
                         f"Allowed values are: {', '.join(allowed_formulations)}.")

    allowed_elements = [4, 9]
    if window_element not in allowed_elements:
        raise ValueError(f"Invalid strain window element type: Q{window_element}. "
                         f"Allowed values are: {', '.join(map(str, allowed_elements))}.")

    if window_size % 2 == 0:
        raise ValueError(f"Invalid strain window size: '{window_size}'. Must be an odd number.")

    # Load data if a file path is given
    if isinstance(data, str):
        results = dic_data_import(layout="matrix", data=data,
                                  binary=input_binary, delimiter=input_delimiter)
    elif isinstance(data, DICResults):
        results = data
    else:
        raise TypeError("data must be either a DICResults instance or a file path string.")

    # Extract dimensions from the validated object
    nss_x = results.ss_x.shape[1]
    nss_y = results.ss_y.shape[0]
    nimg = results.u.shape[0]


    # assigning c++ struct vals for save config
    strain_save_conf = dic2dcpp.SaveConfig()
    strain_save_conf.basepath = output_basepath
    strain_save_conf.binary = output_binary
    strain_save_conf.prefix = output_prefix
    strain_save_conf.delimiter = output_delimiter
    strain_save_conf.at_end = output_at_end

    # Call to C++ backend
    dic2dcpp.strain_engine(results.ss_x, results.ss_y,
                           results.u, results.v,
                           nss_x, nss_y, nimg,
                           window_size, window_element, 
                           strain_formulation, results.filenames,
                           strain_save_conf)



# def strain_data_import(data: str = "./",
#                    binary: bool = False,
#                    layout: str = "matrix",
#                    delimiter: str = " ") -> StrainResults:
#     print("test")
