# ================================================================================
# pyvale: the python validation engine
# License: MIT
# Copyright (C) 2025 The Computer Aided Validation Team
# ================================================================================



import numpy as np
import glob
import os
from pathlib import Path

# Pyvale modules
from pyvale.dic.dicresults import Results

"""
Module responsible for handling importing of DIC results from completed
calculations.
"""


def import_2d(data: str | Path,
              binary: bool = False,
              layout: str = "matrix",
              delimiter: str = ",") -> Results:
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

    # Stack results (except ss_x and ss_y) into arrays
    arrays = [np.stack([frame[i] for frame in frames]) for i in range(len(fields))]

    if layout == "matrix":

        # convert x and y data to meshgrid
        x_unique = np.unique(ss_x_ref)
        y_unique = np.unique(ss_y_ref)
        X, Y = np.meshgrid(x_unique, y_unique)
        shape = (len(files), len(y_unique), len(x_unique))


        arrays = [to_grid(a,shape,ss_x_ref, ss_y_ref, x_unique,y_unique) for a in arrays]


        # sorting out shape function parameters if they are present in the files
        current_shape = arrays[0].shape  # (file,x,y)
        shape_params = np.zeros(())

        # rigid
        if len(fields) == 10:
            shape_params = np.zeros(current_shape+(2,))
            shape_params[:,:,:,0] = arrays[8]
            shape_params[:,:,:,1] = arrays[9]
        if len(fields) == 14:
            shape_params = np.zeros(current_shape+(6,))
            shape_params[:,:,:,0] = arrays[8]
            shape_params[:,:,:,1] = arrays[9]
            shape_params[:,:,:,2] = arrays[10]
            shape_params[:,:,:,3] = arrays[11]
            shape_params[:,:,:,4] = arrays[12]
            shape_params[:,:,:,5] = arrays[13]




        return Results(X, Y, arrays[0], arrays[1], arrays[2], arrays[3],
                          arrays[4], arrays[5], arrays[6], arrays[7], 
                          shape_params, filenames)
    # column layout
    else:

        shape_params = np.zeros(())
        current_shape = arrays[0].shape # (file,(x,y))
        # rigid
        if len(fields) == 10:
            shape_params = np.zeros(current_shape+(2,))
            shape_params[:,:,0] = arrays[8]
            shape_params[:,:,1] = arrays[9]
        if len(fields) == 14:
            shape_params = np.zeros(current_shape+(6,))
            shape_params[:,:,0] = arrays[8]
            shape_params[:,:,1] = arrays[9]
            shape_params[:,:,2] = arrays[10]
            shape_params[:,:,3] = arrays[11]
            shape_params[:,:,4] = arrays[12]
            shape_params[:,:,5] = arrays[13]

        return Results(ss_x_ref, ss_y_ref, arrays[0], arrays[1], arrays[2], arrays[3],
                          arrays[4], arrays[5], arrays[6], arrays[7], 
                          shape_params, filenames)





def read_binary(file: str, delimiter: str):
    """
    Read a binary DIC result file and extract DIC fields.

    Assumes a fixed binary structure with each row containing:
    - 2 × int32 (subset coordinates)
    - 6 × float64 (u, v, match quality, cost, ftol, xtol)
    - 1 × int32 (number of iterations)
    - 1 × uint8 (convergence flag) 
    - 2 or 6 × float64 (shape parameters)

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
    
    # row size can either be 3×4 + 6×8 + 1 = 61 bytes (without shape params)
    # or 3×4 + 6×8 + 1 + 6×8 = 109 bytes (with shape params)
    with open(file, "rb") as f:
        raw = f.read()

    has_shape_params = False
    has_rigid_params = False
    has_affine_params = False
    has_quad_params = False

    row_size_basic = 3 * 4 + 6 * 8 + 1           # 61 bytes
    row_size_with_rigid = row_size_basic + 2 * 8 # 77 bytes
    row_size_with_affine = row_size_basic + 6 * 8 # 109 bytes
    row_size_with_quad = row_size_basic + 12 * 8 # 157 bytes

    if len(raw) % row_size_basic == 0:
        row_size = row_size_basic
        has_shape_params = False
    elif len(raw) % row_size_with_rigid == 0:
        has_shape_params = True
        row_size = row_size_with_rigid
        has_rigid_params = True
    elif len(raw) % row_size_with_affine == 0:
        has_shape_params = True
        row_size = row_size_with_affine
        has_affine_params = True
    elif len(raw) % row_size_with_quad == 0:
        has_shape_params = True
        row_size = row_size_with_quad
        has_quad_params = True
    else:
        raise ValueError(
            f"Binary file has incomplete rows: {file}. "
            f"Expected row size: 65 ((without shape params), "
            f"81 (with rigid shape params) bytes, "
            f"109 (with affine shape params). "
            f"157 (with quad shape params). "
            f"Actual size: {len(raw)} bytes."
        )

    rows = len(raw) // row_size
    arr = np.frombuffer(raw, dtype=np.uint8).reshape(rows, row_size)

    def extract(col, dtype, start):
        return np.frombuffer(arr[:, start:start+col].copy(), dtype=dtype)

    ss_x  = extract(4, np.int32, 0)
    ss_y  = extract(4, np.int32, 4)
    u     = extract(8, np.float64, 8)
    v     = extract(8, np.float64, 16)
    m     = extract(8, np.float64, 24)
    conv  = extract(1, np.uint8, 32).astype(bool)
    cost  = extract(8, np.float64, 33)
    ftol  = extract(8, np.float64, 41)
    xtol  = extract(8, np.float64, 49)
    niter = extract(4, np.int32, 57)

    if has_shape_params:
        if has_rigid_params:
            p0 = extract(8, np.float64, 61)
            p1 = extract(8, np.float64, 69)
            return ss_x, ss_y, u, v, m, conv, cost, ftol, xtol, niter, p0,p1
        if has_affine_params:
            p0 = extract(8, np.float64, 61)
            p1 = extract(8, np.float64, 69)
            p2 = extract(8, np.float64, 77)
            p3 = extract(8, np.float64, 85)
            p4 = extract(8, np.float64, 93)
            p5 = extract(8, np.float64, 101)
            return ss_x, ss_y, u, v, m, conv, cost, ftol, xtol, niter,p0,p1,p2,p3,p4,p5
        if has_quad_params:
            p0 = extract(8, np.float64, 61)
            p1 = extract(8, np.float64, 69)
            p2 = extract(8, np.float64, 77)
            p3 = extract(8, np.float64, 85)
            p4 = extract(8, np.float64, 93)
            p5 = extract(8, np.float64, 101)
            p6 = extract(8, np.float64, 109)
            p7 = extract(8, np.float64, 117)
            p8 = extract(8, np.float64, 125)
            p9 = extract(8, np.float64, 133)
            p10 = extract(8, np.float64, 141)
            p11 = extract(8, np.float64, 149)
            return ss_x, ss_y, u, v, m, conv, cost, ftol, xtol, niter,p0,p1,p2,p3,p4,p5,p6,p7,p8,p9,p10,p11
    else:
        return ss_x, ss_y, u, v, m, conv, cost, ftol, xtol, niter




def read_text(file: str, delimiter: str):
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
        (ss_x, ss_y, u, v, m, conv, cost, ftol, xtol, niter, shape_params)

    Raises
    ------
    ValueError
        If the text file has fewer than 9 columns.
    """

    data = np.loadtxt(file, delimiter=delimiter, skiprows=1)
    
    if data.shape[1] < 9:
        raise ValueError("Text data must have at least 9 columns.")
    
    if data.shape[1] == 10:
        return (
            data[:, 0].astype(np.int32),  # ss_x
            data[:, 1].astype(np.int32),  # ss_y
            data[:, 2], data[:, 3], data[:, 4], # u, v, mag
            data[:, 5].astype(np.bool_), # convergence
            data[:, 6], data[:, 7], data[:,8], # cost, ftol, xtol
            data[:, 9].astype(np.int32) #niter
        )
    #rigid
    elif data.shape[1]==12:
        return (
            data[:, 0].astype(np.int32),  # ss_x
            data[:, 1].astype(np.int32),  # ss_y
            data[:, 2], data[:, 3], data[:, 4], # u, v, mag
            data[:, 5].astype(np.bool_), # convergence
            data[:, 6], data[:, 7], data[:,8], # cost, ftol, xtol
            data[:, 9].astype(np.int32), #niter
            data[:,10], data[:,11] # shape params (rigid)
        )
    #affine
    elif data.shape[1]==16:
        return (
            data[:, 0].astype(np.int32),  # ss_x
            data[:, 1].astype(np.int32),  # ss_y
            data[:, 2], data[:, 3], data[:, 4], # u, v, mag
            data[:, 5].astype(np.bool_), # convergence
            data[:, 6], data[:, 7], data[:,8], # cost, ftol, xtol
            data[:, 9].astype(np.int32), #niter
            data[:,10], data[:,11], data[:,12], data[:,13], data[:,14], data[:,15] # shape params (affine)
        )
    #quad
    elif data.shape[1]==22:
        return (
            data[:, 0].astype(np.int32),  # ss_x
            data[:, 1].astype(np.int32),  # ss_y
            data[:, 2], data[:, 3], data[:, 4], # u, v, mag
            data[:, 5].astype(np.bool_), # convergence
            data[:, 6], data[:, 7], data[:,8], # cost, ftol, xtol
            data[:, 9].astype(np.int32), #niter
            data[:,10], data[:,11], data[:,12], data[:,13], data[:,14], data[:,15],
            data[:,16], data[:,17], data[:,18], data[:,19], data[:,20], data[:,21] # shape params (quad)
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
