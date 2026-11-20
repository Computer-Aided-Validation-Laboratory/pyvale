# ================================================================================
# pyvale: the python validation engine
# License: MIT
# Copyright (C) 2025 The Computer Aided Validation Team
# ================================================================================

import glob
from pathlib import Path
from typing import Literal

import numpy as np

from pyvale.strain.strainresults import StrainResults
from pyvale.dic.dicenum import ELayout
from pyvale.dic.dicimport2d import align_frames_by_coordinates, check_delimiter, to_grid


def import_2d(data: str | Path | list[Path],
              binary: bool = False,
              layout: Literal["COLUMN", "MATRIX", "column", "matrix"] | ELayout = ELayout.MATRIX,
              delimiter: str = ",") -> StrainResults:
    """Import 2D strain result data from text or binary files.

    The importer expects the current format containing the two trailing
    in-plane principal-strain columns, ``eps1`` and ``eps2``. Files also
    contain ``x_mm``, ``y_mm`` and ``z_mm`` after the window pixel coordinates.
    """

    return _import(data, binary=binary, layout=layout, delimiter=delimiter, require_coords=False)


def import_3d(data: str | Path | list[Path],
              binary: bool = False,
              layout: Literal["COLUMN", "MATRIX", "column", "matrix"] | ELayout = ELayout.MATRIX,
              delimiter: str = ",") -> StrainResults:
    """Import 3D-aware strain result data from text or binary files.

    Returned ``StrainResults`` include ``x_mm``, ``y_mm`` and ``z_mm`` arrays for
    the strain-window centre in the cam0 coordinate system.
    """

    return _import(data, binary=binary, layout=layout, delimiter=delimiter, require_coords=True)


def _import(data: str | Path | list[Path],
            binary: bool,
            layout: Literal["COLUMN", "MATRIX", "column", "matrix"] | ELayout,
            delimiter: str,
            require_coords: bool) -> StrainResults:

    layout = ELayout(layout)

    print("Attempting Strain Data import...")

    if isinstance(data, Path):
        data = str(data)

    if isinstance(data, list):
        files = list(map(str, data))
    else:
        files = sorted(glob.glob(data))

    if not files:
        raise FileNotFoundError(f"No results found in: {data}")

    print(f"Found {len(files)} files containing Strain results:")
    for file in files:
        print(f"  - {file}")
    print("")

    def read_file(file: str):
        if binary:
            return read_binary(file, delimiter=delimiter, require_coords=require_coords)
        return read_text(file, delimiter=delimiter)

    # A failed/non-converged subset is omitted from some output files.  Do not
    # stack rows before aligning them: the number and order of rows can differ
    # between frames.  Align all fields by the (x, y) subset coordinate and
    # leave missing values as NaN.
    read_frames = [read_file(file) for file in files]
    coordinate_frames = [(frame[0], frame[1]) for frame in read_frames]
    nfields = len(read_frames[0]) - 2
    if any(len(frame) - 2 != nfields for frame in read_frames):
        raise ValueError("Strain result files contain different numbers of columns.")

    window_x_ref, window_y_ref, arrays = align_frames_by_coordinates(
        coordinate_frames,
        [list(frame[2:]) for frame in read_frames],
    )

    if len(arrays) != 15:
        raise ValueError( "Strain data must be in 17-column format.")

    x_mm, y_mm, z_mm = arrays[:3]
    tensor_arrays = arrays[3:]
    principal_arrays = tensor_arrays[10:12]
    tensor_arrays = tensor_arrays[:10]

    if len(tensor_arrays) != 10:
        raise ValueError(f"Strain data must contain 10 deformation-gradient and strain tensor columns. Number of cols = {len(tensor_arrays)}")

    if layout == ELayout.MATRIX:
        x_unique = np.unique(window_x_ref)
        y_unique = np.unique(window_y_ref)
        window_x_out, window_y_out = np.meshgrid(x_unique, y_unique)
        shape = (len(files), len(y_unique), len(x_unique))
        x_indices = np.searchsorted(x_unique, window_x_ref)
        y_indices = np.searchsorted(y_unique, window_y_ref)

        def grid_per_frame(values):
            output = np.full(shape, np.nan, dtype=np.float64)
            for frame_index in range(len(files)):
                output[frame_index, y_indices, x_indices] = values[frame_index]
            return output

        tensor_arrays = [grid_per_frame(a) for a in tensor_arrays]
        principal_arrays = [grid_per_frame(a) for a in principal_arrays]
        x_mm = grid_per_frame(x_mm)
        y_mm = grid_per_frame(y_mm)
        z_mm = grid_per_frame(z_mm)
    else:
        window_x_out = window_x_ref
        window_y_out = window_y_ref

    return StrainResults(
        window_x=window_x_out,
        window_y=window_y_out,
        def_00=tensor_arrays[0],
        def_01=tensor_arrays[1],
        def_10=tensor_arrays[2],
        def_11=tensor_arrays[3],
        def_20=tensor_arrays[4],
        def_21=tensor_arrays[5],
        eps_xx=tensor_arrays[6],
        eps_xy=tensor_arrays[7],
        eps_yx=tensor_arrays[8],
        eps_yy=tensor_arrays[9],
        filenames=files,
        eps1=principal_arrays[0],
        eps2=principal_arrays[1],
        x_mm=x_mm,
        y_mm=y_mm,
        z_mm=z_mm,
    )


def read_binary(file: str, delimiter: str, require_coords: bool | None = None):
    """Read the current binary strain format.

    Each row contains two window indices, three physical coordinates, six
    deformation-gradient values, four tensor-strain values, and eps1/eps2.
    """

    del delimiter, require_coords

    row_size = 2 * 4 + 3 * 8 + 12 * 8
    with open(file, "rb") as f:
        raw = f.read()

    if len(raw) % row_size != 0:
        raise ValueError(
            f"Binary strain file has incomplete rows: {file}. "
            f"Expected row size {row_size}, got {len(raw)} bytes."
        )

    rows = len(raw) // row_size
    arr = np.frombuffer(raw, dtype=np.uint8).reshape(rows, row_size)

    def extract(width, dtype, start):
        return np.frombuffer(arr[:, start:start + width].copy(), dtype=dtype)

    offset = 0
    window_x = extract(4, np.int32, offset); offset += 4
    window_y = extract(4, np.int32, offset); offset += 4
    coord_arrays = [
        extract(8, np.float64, offset),
        extract(8, np.float64, offset + 8),
        extract(8, np.float64, offset + 16),
    ]
    offset += 24

    tensor_arrays = []
    for _ in range(12):
        tensor_arrays.append(extract(8, np.float64, offset))
        offset += 8

    return (window_x, window_y, *coord_arrays, *tensor_arrays)


def read_text(file: str, delimiter: str):
    """Read a text strain result file.

    The current text format contains 17 columns, including three physical
    coordinates and the final ``eps1`` and ``eps2`` columns.

    """

    check_delimiter(file, delimiter)
    data = np.loadtxt(file, delimiter=delimiter, skiprows=1)
    if data.ndim == 1:
        data = data.reshape(1, -1)

    if data.shape[1] != 17:
        raise ValueError(f"Text strain data must contain 17 columns, got {data.shape[1]}")

    return (
        data[:, 0].astype(np.int32),
        data[:, 1].astype(np.int32),
        *[data[:, i] for i in range(2, data.shape[1])],
    )
