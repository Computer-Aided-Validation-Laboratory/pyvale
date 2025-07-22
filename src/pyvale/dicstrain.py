# ================================================================================
# pyvale: the python validation engine
# License: MIT
# Copyright (C) 2025 The Computer Aided Validation Team
# ================================================================================

import matplotlib.pyplot as plt
import numpy as np
import glob
from pathlib import Path

from pyvale import dic2dcpp
from pyvale import dicchecks
from pyvale.dicdataimport import dic_data_import
from pyvale.dicresults import DICResults
from pyvale.dicstrainresults import StrainResults

def strain_2d(data: str | Path,
              window_size: int=5, 
              window_element: int=4,
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

    filenames = dicchecks.check_strain_files(strain_files=data)

    # Load data if a file path is given
    results = dic_data_import(layout="matrix", data=str(data),
                                  binary=input_binary, delimiter=input_delimiter)

    # Extract dimensions from the validated object
    nss_x = results.ss_x.shape[1]
    nss_y = results.ss_y.shape[0]
    nimg = results.u.shape[0]


    dicchecks.check_output_directory(str(output_basepath), output_prefix)

    # assigning c++ struct vals for save config
    strain_save_conf = dic2dcpp.SaveConfig()
    strain_save_conf.basepath = str(output_basepath)
    strain_save_conf.binary = output_binary
    strain_save_conf.prefix = output_prefix
    strain_save_conf.delimiter = output_delimiter
    strain_save_conf.at_end = output_at_end

    print(filenames)

    # Call to C++ backend
    dic2dcpp.strain_engine(results.ss_x, results.ss_y,
                           results.u, results.v,
                           nss_x, nss_y, nimg,
                           window_size, window_element, 
                           strain_formulation, filenames,
                           strain_save_conf)







def strain_data_import(data: str | Path,
                   binary: bool = False,
                   layout: str = "matrix",
                   delimiter: str = " ") -> StrainResults:
    """
    Import strain result data from human readable text or binary files.

    Parameters
    ----------

    data : str or pathlib.Path
        Path pattern to the data files (can include wildcards). Default is "./".

    layout : str, optional
        Format of the output data layout: "column" (flat array per frame) or "matrix" 
        (reshaped grid per frame). Default is "column".

    binary : bool, optional
        If True, expects files in a specific binary format. If False, expects text data. 
        Default is False.

    delimiter : str, optional
        Delimiter used in text data files. Ignored if binary=True. Default is a single space.

    Returns
    -------
    StrainResults
        A named container with the following fields:
            - window_x, window_y (grid arrays if layout=="matrix"; otherwise, 1D integer arrays)
            - def_grad, eps (deformation gradient and strain arrays with shape depending on layout)
            - filenames (python list)

    Raises
    ------
    ValueError:
        If `layout` is not "column" or "matrix", or text data has insufficient columns,
        or binary rows are malformed.
        
    FileNotFoundError:
        If no matching data files are found.
    """




    print("")
    print("Attempting Strain Data import...")
    print("")
    
    # convert to str 
    if isinstance(data, Path):
        data = str(data)


    files = sorted(glob.glob(data))
    filenames = files
    if not files:
        raise FileNotFoundError(f"No results found in: {data}")

    print(f"Found {len(files)} files containing Strain results:")
    for file in files:
        print(f"  - {file}")
    print("")


    # Read first file to define reference coordinates
    read_data = read_binary if binary else read_text

    window_x_ref, window_y_ref, *fields = read_data(files[0], delimiter=delimiter)
    frames = [list(fields)]

    for file in files[1:]:
        window_x, window_y, *f = read_data(file, delimiter)
        if not (np.array_equal(window_x_ref, window_x) and
                np.array_equal(window_y_ref, window_y)):
            raise ValueError("Mismatch in coordinates across frames.")
        frames.append(f)

    # Stack fields into arrays
    arrays = [np.stack([frame[i] for frame in frames]) for i in range(8)]

    # if reading into a matrix layout we need to convert to meshgrids
    if layout == "matrix":

        # create meshgrid
        x_unique = np.unique(window_x_ref)
        y_unique = np.unique(window_y_ref)
        X, Y = np.meshgrid(x_unique, y_unique)

        # convert results to a grid based on meshgrid dims
        shape = (len(files), len(y_unique), len(x_unique))
        arrays = [to_grid(a,shape,window_x_ref, window_y_ref, x_unique,y_unique) for a in arrays]


        # combine strain results into single np.ndarray. current dimeensions of
        # each array are (file,x,y). The results will become
        # (file,x,y,matrix_x,def_matrix_y)
        current_shape = arrays[0].shape # (file,x,y)
        def_grad = np.zeros(current_shape+(2,2))
        eps = np.zeros(current_shape+(2,2))


        def_grad[:,:,:,0,0] = arrays[0]
        def_grad[:,:,:,0,1] = arrays[1]
        def_grad[:,:,:,1,0] = arrays[2]
        def_grad[:,:,:,1,1] = arrays[3]
        eps[:,:,:,0,0] = arrays[4]
        eps[:,:,:,0,1] = arrays[5]
        eps[:,:,:,1,0] = arrays[6]
        eps[:,:,:,1,1] = arrays[7]

        return StrainResults(X, Y, def_grad, eps, filenames)

    else:
        current_shape = arrays[0].shape # (file,(x,y))
        def_grad = np.zeros(current_shape+(2,2))
        eps = np.zeros(current_shape+(2,2))
        def_grad[:,:,0,0] = arrays[0]
        def_grad[:,:,0,1] = arrays[1]
        def_grad[:,:,1,0] = arrays[2]
        def_grad[:,:,1,1] = arrays[3]
        eps[:,:,0,0] = arrays[4]
        eps[:,:,0,1] = arrays[5]
        eps[:,:,1,0] = arrays[6]
        eps[:,:,1,1] = arrays[7]
        return StrainResults(window_x_ref, window_y_ref, def_grad, eps, filenames)



def read_binary(file: str, delimiter: str):
    """
    Read a binary Strain result file and extract DIC fields.

    Assumes a fixed binary structure with each row containing:
    - 2x int32 (subset coordinates)
    - 8x float64 (deformation matrix, strain matrix)

    Parameters
    ----------
    file : str
        Path to the binary result file.

    delimiter : str
        Ignored for binary data (included for API consistency).

    Returns
    -------
    tuple of np.ndarray
        Arrays corresponding to:
        (window_x, window_y, def_grad, eps)

    Raises
    ------
    ValueError
        If the binary file size does not align with expected row size.
    """

    row_size = (3 * 4 + 6 * 8)
    with open(file, "rb") as f:
        raw = f.read()
    if len(raw) % row_size != 0:
        raise ValueError("Binary file has incomplete rows.")

    rows = len(raw) // row_size
    arr = np.frombuffer(raw, dtype=np.uint8).reshape(rows, row_size)
    
    # lil function to make extracting a bit easier
    def extract(col, dtype, start): 
        return np.frombuffer(arr[:, start:start+col], dtype=dtype)

    window_x = extract(4, np.int32, 0)
    window_y = extract(4, np.int32, 4)
    def_grad00 = extract(8, np.float64, 8)
    def_grad01 = extract(8, np.float64, 16)
    def_grad10 = extract(8, np.float64, 24)
    def_grad11 = extract(8, np.float64, 32)
    eps00 = extract(8, np.float64, 40)
    eps01 = extract(8, np.float64, 48)
    eps10 = extract(8, np.float64, 56)
    eps11 = extract(8, np.float64, 72)

    return window_x, window_y, def_grad00, def_grad01, def_grad10, def_grad11, eps00, eps01, eps10, eps11




def read_text(file: str, delimiter: str):
    """
    Read a human-readable text DIC result file and extract DIC fields.

    Expects at least 9 columns:
    [ss_x, ss_y, u, v, m, cost, ftol, xtol, niter]

    Parameters
    ----------
    file : str
        Path to the text result file.

    delimiter : str
        Delimiter used in the text file (e.g., space, tab, comma).

    Returns
    -------
    tuple of np.ndarray
        Arrays corresponding to:
        (ss_x, ss_y, u, v, m, cost, ftol, xtol, niter)

    Raises
    ------
    ValueError
        If the text file has fewer than 9 columns.
    """

    data = np.loadtxt(file, delimiter=delimiter, skiprows=1)
    if data.shape[1] < 9:
        raise ValueError("Text data must have at least 9 columns.")
    return (
        data[:, 0].astype(np.int32), # window_x
        data[:, 1].astype(np.int32), # window_y
        data[:, 2], data[:, 3], data[:, 4], data[:, 5], #def_grad
        data[:, 6], data[:, 7], data[:, 8], data[:, 9]  #eps
    )






def to_grid(data, shape, ss_x_ref, ss_y_ref, x_unique, y_unique):
    """
    Reshape a 2D DIC field from flat (column) format into grid (matrix) format.

    This is used when output layout is specified as "matrix".
    Maps values using reference subset coordinates (ss_x_ref, ss_y_ref).

    Parameters
    ----------
    data : np.ndarray
        Array of shape (n_frames, n_points) to be reshaped into (n_frames, height, width).

    shape : tuple
        Target shape of output array: (n_frames, height, width).

    ss_x_ref : np.ndarray
        X coordinates of subset centers.

    ss_y_ref : np.ndarray
        Y coordinates of subset centers.

    x_unique : np.ndarray
        Sorted unique X coordinates in the grid.

    y_unique : np.ndarray
        Sorted unique Y coordinates in the grid.

    Returns
    -------
    np.ndarray
        Reshaped array with shape `shape`, filled with NaNs where no data exists.
    """

    grid = np.full(shape, np.nan)
    for i, (x, y) in enumerate(zip(ss_x_ref, ss_y_ref)):
        x_idx = np.where(x_unique == x)[0][0]
        y_idx = np.where(y_unique == y)[0][0]
        grid[:, y_idx, x_idx] = data[:, i]
    return grid
