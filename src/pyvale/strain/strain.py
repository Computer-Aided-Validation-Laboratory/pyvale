# ================================================================================
# pyvale: the python validation engine
# License: MIT
# Copyright (C) 2025 The Computer Aided Validation Team
# ================================================================================

from pathlib import Path

# pyvale
from pyvale.strain.strainresults import StrainResults
from pyvale.strain.strainchecks import check_strain_files
from pyvale.dic.dicdataimport import import_2d
from pyvale.common_py.util import check_output_directory
import pyvale.strain.strain_cpp as strain_cpp
import pyvale.common_cpp.common_cpp as common_cpp

def calculate_2d(data: str | Path,
              window_size: int=5, 
              window_element: int=9,
              input_binary: bool=False,
              input_delimiter: str=",",
              output_basepath: Path | str="./",
              output_binary: bool=False,
              output_prefix: str="strain_",
              output_delimiter: str=",",
              output_at_end: bool=False,
              strain_formulation: str="HENCKY"):
    """
    Compute strain fields from DIC displacement data using a finite element smoothing approach.

    This function validates the input data and parameters, optionally loads DIC results from file,
    and passes the data to a C++-accelerated backend for strain computation.

    Parameters
    ----------
    data : pathlib.Path or str
        A pathlib.Path or str to files from which the data should be imported.
    input_delimiter: str
        delimiter used for the input dic results files (default: ",").
    input_binary bool:
        whether input data is in human-readable or binary format (default:
        False).
    window_size : int, optional
        The size of the local window over which to compute strain (must be odd), by default 5.
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
        Prefix for all output files (default: "strain_"). results will be
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

    filenames = check_strain_files(strain_files=data)

    # Load data if a file path is given
    results = import_2d(layout="matrix", data=str(data),
                                  binary=input_binary, delimiter=input_delimiter)

    # Extract dimensions from the validated object
    nss_x = results.ss_x.shape[1]
    nss_y = results.ss_y.shape[0]
    nimg = results.u.shape[0]


    check_output_directory(str(output_basepath), output_prefix, 0)

    # assigning c++ struct vals for save config
    strain_save_conf = common_cpp.SaveConfig()
    strain_save_conf.basepath = str(output_basepath)
    strain_save_conf.binary = output_binary
    strain_save_conf.prefix = output_prefix
    strain_save_conf.delimiter = output_delimiter
    strain_save_conf.at_end = output_at_end

    print(filenames)

    # Call to C++ backend
    strain_cpp.strain_engine(results.ss_x, results.ss_y,
                           results.u, results.v,
                           nss_x, nss_y, nimg,
                           window_size, window_element, 
                           strain_formulation, filenames,
                           strain_save_conf)





