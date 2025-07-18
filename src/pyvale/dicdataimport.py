# ================================================================================
# pyvale: the python validation engine
# License: MIT
# Copyright (C) 2025 The Computer Aided Validation Team
# ================================================================================



import numpy as np
import glob
import os
from pathlib import Path

# import cython module
from pyvale.dicresults import DICResults

"""
Module responsible for handling importing of DIC results from completed
calculations.
"""


def dic_data_import(data: str | Path,
                   binary: bool = False,
                   layout: str = "matrix",
                   delimiter: str = " ") -> DICResults:
    """
    Import DIC result data from human readable text or binary files.

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
    DICResults
        A named container with the following fields:
            - ss_x, ss_y (grid arrays if layout=="matrix"; otherwise, 1D integer arrays)
            - u, v, m, converged, cost, ftol, xtol, niter (arrays with shape depending on layout)
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
    print("Attempting DIC Data import...")
    print("")

    # convert to str 
    if isinstance(data, Path):
        data = str(data)

    files = sorted(glob.glob(data))
    filenames = files
    if not files:
        raise FileNotFoundError(f"No results found in: {data}")

    print(f"Found {len(files)} files containing DIC results:")
    for file in files:
        print(f"  - {file}")
    print("")


    # Read first file to define reference coordinates
    read_data = read_binary if binary else read_text
    ss_x_ref, ss_y_ref, *fields = read_data(files[0], delimiter=delimiter)
    frames = [list(fields)]

    for file in files[1:]:
        ss_x, ss_y, *f = read_data(file, delimiter)
        if not (np.array_equal(ss_x_ref, ss_x) and np.array_equal(ss_y_ref, ss_y)):
            raise ValueError("Mismatch in coordinates across frames.")
        frames.append(f)

    # Stack fields into arrays
    arrays = [np.stack([frame[i] for frame in frames]) for i in range(8)]

    if layout == "matrix":
        x_unique = np.unique(ss_x_ref)
        y_unique = np.unique(ss_y_ref)
        X, Y = np.meshgrid(x_unique, y_unique)
        shape = (len(files), len(y_unique), len(x_unique))
        arrays = [to_grid(a,shape,ss_x_ref, ss_y_ref, x_unique,y_unique) for a in arrays]
        return DICResults(X, Y, *arrays, filenames)
    else:
        return DICResults(ss_x_ref, ss_y_ref, *arrays, filenames)





def read_binary(file: str, delimiter: str):
    """
    Read a binary DIC result file and extract DIC fields.

    Assumes a fixed binary structure with each row containing:
    - 2 × int32 (subset coordinates)
    - 6 × float64 (u, v, match quality, cost, ftol, xtol)
    - 1 × int32 (number of iterations)

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
        (ss_x, ss_y, u, v, m, cost, ftol, xtol, niter)

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
    def extract(col, dtype, start): return np.frombuffer(arr[:, start:start+col], dtype=dtype)
    ss_x = extract(4, np.int32, 0)
    ss_y = extract(4, np.int32, 4)
    u    = extract(8, np.float64, 8)
    v    = extract(8, np.float64, 16)
    m    = extract(8, np.float64, 24)
    conv = extract(1, np.bool_, 25)
    cost = extract(8, np.float64, 33)
    ftol = extract(8, np.float64, 41)
    xtol = extract(8, np.float64, 49)
    niter = extract(4, np.int32, 57)
    return ss_x, ss_y, u, v, m, conv, cost, ftol, xtol, niter




def read_text(file: str, delimiter: str):
    """
    Read a human-readable text DIC result file and extract DIC fields.

    Expects at least 9 columns:
    [ss_x, ss_y, u, v, m, conv, cost, ftol, xtol, niter]

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
        (ss_x, ss_y, u, v, m, conv, cost, ftol, xtol, niter)

    Raises
    ------
    ValueError
        If the text file has fewer than 9 columns.
    """

    data = np.loadtxt(file, delimiter=delimiter, skiprows=1)
    if data.shape[1] < 9:
        raise ValueError("Text data must have at least 9 columns.")
    return (
        data[:, 0].astype(np.int32),  # ss_x
        data[:, 1].astype(np.int32),  # ss_y
        data[:, 2], data[:, 3], data[:, 4], # u, v, mag
        data[:, 5].astype(np.bool_), # convergence
        data[:, 6], data[:, 7], data[:,8], # cost, ftol, xtol
        data[:, 9].astype(np.int32) #niter
    )





def to_grid(data, shape, ss_x_ref, ss_y_ref, x_unique, y_unique):
    """
    Reshape a 2D DIC field from flat (column) format into grid (matrix) format.

    This is used when output layout is specified as "matrix".
    Maps values using reference subset coordinates (ss_x_ref, ss_y_ref).

    Parameters
    ol
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
