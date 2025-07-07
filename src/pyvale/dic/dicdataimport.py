# ================================================================================
# pyvale: the python validation engine
# License: MIT
# Copyright (C) 2025 The Computer Aided Validation Team
# ================================================================================



import numpy as np
import glob
from typing import Union
from typing import List

# import cython module
from pyvale.dic.dicresults import DICResults


def DICdata_import(layout: str = "column", 
                   data: str = "./", 
                   binary: bool = False,
                   delimiter: str = " ") -> DICResults:
    """
    Import DIC result data from human readable text or binary files.

    Parameters:
    -----------
    layout : str, optional
        Format of the output data layout: "column" (flat array per frame) or "matrix" 
        (reshaped grid per frame). Default is "column".
        
    data : str, optional
        Path pattern to the data files (can include wildcards). Default is "./".
        
    binary : bool, optional
        If True, expects files in a specific binary format. If False, expects text data. 
        Default is False.
        
    delimiter : str, optional
        Delimiter used in text data files. Ignored if binary=True. Default is a single space.

    Returns:
    --------
    DICResults
        A named container with the following fields:
            - X, Y (grid arrays if layout=="matrix"; otherwise, 1D integer arrays)
            - u, v, m, cost, ftol, xtol, niter (arrays with shape depending on layout)

    Raises:
    -------
    ValueError:
        If `layout` is not "column" or "matrix", or text data has insufficient columns,
        or binary rows are malformed.
        
    FileNotFoundError:
        If no matching data files are found.
    """

    files = sorted(glob.glob(data))
    if not files:
        raise FileNotFoundError(f"No results found in: {data}")

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
    arrays = [np.stack([frame[i] for frame in frames]) for i in range(7)]

    if layout == "matrix":
        x_unique = np.unique(ss_x_ref)
        y_unique = np.unique(ss_y_ref)
        X, Y = np.meshgrid(x_unique, y_unique)
        shape = (len(files), len(y_unique), len(x_unique))



        arrays = [to_grid(a,shape,ss_x_ref, ss_y_ref, x_unique,y_unique) for a in arrays]
        return DICResults(X, Y, *arrays)
    else:
        return DICResults(ss_x_ref, ss_y_ref, *arrays)



def read_binary(file: str, delimiter: str):
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
    cost = extract(8, np.float64, 32)
    ftol = extract(8, np.float64, 40)
    xtol = extract(8, np.float64, 48)
    niter = extract(4, np.int32, 56)
    return ss_x, ss_y, u, v, m, cost, ftol, xtol, niter

def read_text(file: str, delimiter: str):
    data = np.loadtxt(file, delimiter=delimiter)
    if data.shape[1] < 9:
        raise ValueError("Text data must have at least 9 columns.")
    return (
        data[:, 0].astype(np.int32),  # ss_x
        data[:, 1].astype(np.int32),  # ss_y
        data[:, 2], data[:, 3], data[:, 4],
        data[:, 5], data[:, 6], data[:, 7],
        data[:, 8].astype(np.int32)
    )

def to_grid(data, shape, ss_x_ref, ss_y_ref, x_unique, y_unique):
    grid = np.full(shape, np.nan)
    for i, (x, y) in enumerate(zip(ss_x_ref, ss_y_ref)):
        x_idx = np.where(x_unique == x)[0][0]
        y_idx = np.where(y_unique == y)[0][0]
        grid[:, y_idx, x_idx] = data[:, i]
    return grid
