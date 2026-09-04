# ================================================================================
# pyvale: the python validation engine
# License: MIT
# Copyright (C) 2025 The Computer Aided Validation Team
# ================================================================================



import numpy as np
import glob
import os
from pathlib import Path
from typing import Literal

# Pyvale modules
from pyvale.dic.dicresults import Results
import pyvale.common_py.util as common_py_util

"""
Module responsible for handling importing of DIC results from completed
calculations.
"""


def import_2d(data: str | Path | list[Path],
              delimiter: str,
              binary: bool = False,
              layout: Literal["column", "matrix"] = "matrix",
              print_level: int=1) -> Results:

    """
    Import DIC result data from human readable text or binary files.

    Parameters
    ----------

    data : str or pathlib.Path or list[pathlib.Path]
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
    Results
        A named container with the following fields:
            - ss_x, ss_y (grid arrays if layout=="matrix"; otherwise, 1D integer arrays)
            - u, v, m, converged, cost, ftol, xtol, niter (arrays with shape depending on layout)
            - filenames (python list)

    Raises
    ------
    ValueError:
        If `layout` is not "column" or "matrix", or text data has insufficient columns,
        or binary rows are malformed.
        import cython module
    FileNotFoundError:
        If no matching data files are found.
    """

    if (print_level>0):
        common_py_util.print_title("Importing DIC Results")

    # convert to str 
    if isinstance(data, Path):
        data = str(data)

    if isinstance(data, list):
        files = list(map(str, data))
    else:
        files = sorted(glob.glob(data))
        if not files:
            raise FileNotFoundError(f"No results found in: {data}")
    
    if print_level>0:
        common_py_util.info_out(f"Found {len(files)} files containing DIC results:", "")
        for file in files:
            common_py_util.info_out(f"{file}", "")


    # Read first file to define reference coordinates
    read_data = read_binary if binary else read_text
    ss_x_ref, ss_y_ref, *fields = read_data(files[0], delimiter=delimiter, print_level=print_level)
    frames = [list(fields)]

    for file in files[1:]:
        ss_x, ss_y, *f = read_data(file, delimiter)
        if not (np.array_equal(ss_x_ref, ss_x) and np.array_equal(ss_y_ref, ss_y)):
            raise ValueError("Mismatch in coordinates across frames.")
        frames.append(f)

    # Stack results (except ss_x and ss_y) into arrays
    arrays = [np.stack([frame[i] for frame in frames]) for i in range(len(fields))]

    if print_level>0:
        common_py_util.info_out(f"Imported {len(files)} frames of DIC data.", "")

    if layout == "matrix":

        # convert x and y data to meshgrid
        x_unique = np.unique(ss_x_ref)
        y_unique = np.unique(ss_y_ref)
        X, Y = np.meshgrid(x_unique, y_unique)
        shape = (len(files), len(y_unique), len(x_unique))

        arrays = [to_grid(a,shape,ss_x_ref, ss_y_ref, x_unique,y_unique) for a in arrays]

        return Results(ss_x=X,
                       ss_y=Y,
                       u_px=arrays[0],
                       v_px=arrays[1],
                       mag_px=arrays[2],
                       converged=arrays[3],
                       cost=arrays[4],
                       ftol=arrays[5],
                       xtol=arrays[6],
                       niter=arrays[7],
                       filenames=files)

    # column layout
    else:

        return Results(ss_x=ss_x_ref, 
                       ss_y=ss_y_ref,
                       u_px=arrays[0],
                       v_px=arrays[1],
                       mag_px=arrays[2],
                       converged=arrays[3],
                       cost=arrays[4],
                       ftol=arrays[5],
                       xtol=arrays[6],
                       niter=arrays[7],
                       filenames=files)


def read_binary(file: str, delimiter: str, print_level: int=1):
    """
    Read a binary 2D DIC result file and extract DIC fields.

    Supports rows written by ``ResultArrays::write_to_disk_2d`` with or without
    optional shape parameters. Shape parameters are currently ignored by the
    public ``Results`` dataclass.
    """

    del delimiter

    if print_level>0:
        common_py_util.info(f"Reading binary DIC result file: {file}")

    with open(file, "rb") as f:
        raw = f.read()

    row_size_basic = 2 * 4 + 3 * 8 + 1 + 3 * 8 + 4
    row_sizes = [row_size_basic + nparams * 8 for nparams in (12, 6, 2, 0)]

    row_size = next((size for size in row_sizes if len(raw) % size == 0), None)
    if row_size is None:
        raise ValueError(
            f"Binary file has incomplete rows: {file}. "
            f"Expected row size one of {row_sizes}, actual size: {len(raw)} bytes."
        )

    rows = len(raw) // row_size
    arr = np.frombuffer(raw, dtype=np.uint8).reshape(rows, row_size)

    def extract(width, dtype, start):
        return np.frombuffer(arr[:, start:start + width].copy(), dtype=dtype)

    offset = 0
    ss_x = extract(4, np.int32, offset); offset += 4
    ss_y = extract(4, np.int32, offset); offset += 4
    u = extract(8, np.float64, offset); offset += 8
    v = extract(8, np.float64, offset); offset += 8
    mag = extract(8, np.float64, offset); offset += 8
    conv = extract(1, np.uint8, offset).astype(bool); offset += 1
    cost = extract(8, np.float64, offset); offset += 8
    ftol = extract(8, np.float64, offset); offset += 8
    xtol = extract(8, np.float64, offset); offset += 8
    niter = extract(4, np.int32, offset)

    return ss_x, ss_y, u, v, mag, conv, cost, ftol, xtol, niter




def read_text(file: str, delimiter: str, print_level: int=1):
    """
    Read a human-readable text DIC result file and extract DIC fields.

    Expects at least 9 columns:
    [ss_x, ss_y, u, v, m, conv, cost, ftol, xtol, niter]
    Could also include shape parameters if present.

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
    if print_level>0:
        common_py_util.info(f"Reading text DIC result file: {file}")

    check_delimiter(file, delimiter)
    data = np.loadtxt(file, delimiter=delimiter, skiprows=1)
    
    if data.shape[1] != 10:
        raise ValueError(f"Input DIC data must have exactly 10 columns. Num cols = {data.shape[1]}")

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

def check_delimiter(fname: str, delimiter: str | None) -> None:
    with open(fname, "r") as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#"):
                first_line = line
                break
        else:
            return

    if delimiter == "," and "," not in first_line:
        raise ValueError(
            f"Expected comma-separated data but first data row contains no commas:\n"
            f"    {first_line[:120]}"
        )

    if delimiter == "\t" and "\t" not in first_line:
        raise ValueError(
            f"Expected tab-separated data but first data row contains no tabs:\n"
            f"    {first_line[:120]}"
        )
