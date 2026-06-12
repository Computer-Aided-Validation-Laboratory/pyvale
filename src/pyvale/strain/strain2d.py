# ================================================================================
# pyvale: the python validation engine
# License: MIT
# Copyright (C) 2025 The Computer Aided Validation Team
# ================================================================================

from pathlib import Path
from typing import Literal
import numpy as np

# pyvale
from pyvale.strain.strainresults import StrainResults
from pyvale.strain.strainchecks import check_strain_files
from pyvale.dic.dicimport2d import import_2d
from pyvale.dic.dicresults import Results as dicResults
from pyvale.common_py.util import check_output_directory
import pyvale.strain.strain_cpp as strain_cpp
import pyvale.common_cpp.common_cpp as common_cpp

def calculate_2d(data: dicResults | str | Path | list[Path],
              window_size: int=5, 
              window_element: int=9,
              input_binary: bool=False,
              input_delimiter: str=",",
              output_basepath: Path | str="./",
              output_binary: bool=False,
              output_prefix: str="strain_",
              output_delimiter: str=",",
              strain_formulation: Literal["GREEN", "ALMANSI", "HENCKY", "BIOT_EULER", "BIOT_LAGRANGE"] = "HENCKY"):
    """
    Compute strain fields from DIC displacement data using a finite element smoothing approach.

    This function validates the input data and parameters, optionally loads DIC results from file,
    and passes the data to a C++-accelerated backend for strain computation.

    Parameters
    ----------
    data : dic.Results, pathlib.Path, list[pathlib.Path] str
        input data can either be a dic.Results object or pathlib.Path / str if importing data 
        straight from a file
    input_delimiter: str
        delimiter used for the input dic results files if using
        pathlib.Path or str for data import (default: ",").
    input_binary bool:
        whether input data is in human-readable or binary format if using
        pathlib.Path or str for data import (default: False).
    window_size : int, optional
        The size of the local window over which to compute strain (must be odd,
        default: 5).
    window_element : int, optional
        The type of finite element shape function used in the strain window: 4 (bilinear) or 9 (biquadratic),
        by default 4.
    strain_formulation : str, optional
        The strain definition to use: one of 'GREEN', 'ALMANSI', 'HENCKY', 'BIOT_EULER', 'BIOT_LAGRANGE'.
        Defaults to 'HENCKY'.
    output_basepath : str or pathlib.Path, optional
        Directory path where output files will be written (default: "./").
    output_binary : bool, optional
        Whether to write output in binary format (default: False).
    output_prefix : str, optional
        Prefix for all output files (default: :code:`strain_`). results will be
        named with output_prefix + original filename. THe extension will be
        changed to ".csv" or ".dic2d" depending on whether outputting as a binary.
    output_delimiter : str, optional
        Delimiter used in text output files (default: ",").

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


    if isinstance(data, (str, Path, list)):
        filenames = check_strain_files(strain_files=data)

        # Load data if a file path is given
        dicresults = import_2d(layout="matrix", data=data, 
                            binary=input_binary, delimiter=input_delimiter)

    elif isinstance(data, dicResults):
        dicresults = data
        print(dicresults.ss_x.shape, dicresults.ss_y.shape, dicresults.u.shape,dicresults.v.shape)
        assert dicresults.ss_x.ndim == 2 and dicresults.ss_y.ndim == 2, "ss_x and ss_y must be 2D"
        assert dicresults.ss_x.shape == dicresults.ss_y.shape, "ss_x and ss_y must have the same shape"
        assert dicresults.u.ndim == 3 and dicresults.v.ndim == 3, "u and v must be 3D"
        assert dicresults.u.shape == dicresults.v.shape, "u and v must have the same shape"
        assert dicresults.u.shape[1:] == dicresults.ss_x.shape, "Spatial dimensions of u must match ss_x"

        # need to make dummy filenames
        filenames = []
        for f in range(0,dicresults.u.shape[0]):
            filenames.append(f"strain_data_{f:04d}")

    else: 
        raise TypeError(f"Unexpected displacement data type: {type(data)}")

    # Extract dimensions from the validated object
    nss_x = dicresults.ss_x.shape[1]
    nss_y = dicresults.ss_x.shape[0]
    nimg = dicresults.u.shape[0]


    check_output_directory(str(output_basepath), output_prefix, 0)

    # assigning c++ struct vals for save config
    strain_save_conf = common_cpp.SaveConfig()
    strain_save_conf.basepath = str(output_basepath)
    strain_save_conf.binary = output_binary
    strain_save_conf.prefix = output_prefix
    strain_save_conf.delimiter = output_delimiter

    print(type(filenames))

    # make an empty array for w 
    w_dummy = np.zeros_like(dicresults.u)

    # Call to C++ backend
    strain_cpp.strain_engine(dicresults.ss_x, dicresults.ss_y,
                           dicresults.u, dicresults.v, w_dummy,
                           nss_x, nss_y, nimg,
                           window_size, window_element, 
                           strain_formulation, filenames,
                           strain_save_conf)





