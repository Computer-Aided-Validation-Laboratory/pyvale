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
from pyvale.dic.dicimport2d import check_delimiter


def import_2d(data: str | Path | list[Path],
              binary: bool = False,
              layout: Literal["COLUMN", "MATRIX", "column", "matrix"] | ELayout = ELayout.MATRIX,
              delimiter: str = ",") -> StrainResults:
    """Import 2D strain result data from text or binary files.

    The importer accepts the legacy 20-column strain format and the 3D-aware
    23-column format that appends ``x_mm``, ``y_mm`` and ``z_mm`` after the
    strain-window pixel coordinates.
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

    coordinates = np.unique(
        np.concatenate(
            [np.column_stack((x, y)) for x, y in coordinate_frames],
            axis=0,
        ),
        axis=0,
    )
    window_x_ref = coordinates[:, 0].astype(np.int32)
    window_y_ref = coordinates[:, 1].astype(np.int32)

    coordinate_index = {
        (int(x), int(y)): index
        for index, (x, y) in enumerate(coordinates)
    }
    arrays = [
        np.full((len(files), len(coordinates)), np.nan, dtype=np.float64)
        for _ in range(nfields)
    ]
    for frame_index, frame in enumerate(read_frames):
        row_indices = [
            coordinate_index[(int(x), int(y))]
            for x, y in zip(frame[0], frame[1])
        ]
        if len(set(row_indices)) != len(row_indices):
            raise ValueError(
                f"Duplicate subset coordinates in strain result file: {files[frame_index]}"
            )
        for field_index, values in enumerate(frame[2:]):
            arrays[field_index][frame_index, row_indices] = values

    has_coords = len(arrays) == 13
    if require_coords and not has_coords:
        raise ValueError("3D strain data must include x_mm, y_mm and z_mm columns.")

    if has_coords:
        x_mm, y_mm, z_mm = arrays[:3]
        tensor_arrays = arrays[3:]
    else:
        x_mm = y_mm = z_mm = None
        tensor_arrays = arrays

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
        if has_coords:
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
        x_mm=x_mm,
        y_mm=y_mm,
        z_mm=z_mm,
    )


def read_binary(file: str, delimiter: str, require_coords: bool | None = None):
    """Read a binary strain result file.

    Supports legacy rows with ``window_x``, ``window_y`` and 18 doubles, plus
    new 3D-aware rows with ``x_mm``, ``y_mm`` and ``z_mm`` before the 18 tensor
    values.
    """

    del delimiter

    row_size_2d = 2 * 4 + 18 * 8
    row_size_3d = 2 * 4 + 3 * 8 + 18 * 8

    with open(file, "rb") as f:
        raw = f.read()

    if require_coords is True:
        if len(raw) % row_size_3d != 0:
            raise ValueError(
                f"Binary 3D strain file has incomplete rows: {file}. "
                f"Expected row size {row_size_3d}, got {len(raw)} bytes."
            )
        row_size = row_size_3d
        has_coords = True
    elif require_coords is False and Path(file).suffix != ".dic3d":
        if len(raw) % row_size_2d != 0:
            raise ValueError(
                f"Binary 2D strain file has incomplete rows: {file}. "
                f"Expected row size {row_size_2d}, got {len(raw)} bytes."
            )
        row_size = row_size_2d
        has_coords = False
    elif len(raw) % row_size_3d == 0:
        row_size = row_size_3d
        has_coords = True
    elif len(raw) % row_size_2d == 0:
        row_size = row_size_2d
        has_coords = False
    else:
        raise ValueError(
            f"Binary file has incomplete rows: {file}. "
            f"Expected row size {row_size_2d} or {row_size_3d}, got {len(raw)} bytes."
        )

    rows = len(raw) // row_size
    arr = np.frombuffer(raw, dtype=np.uint8).reshape(rows, row_size)

    def extract(width, dtype, start):
        return np.frombuffer(arr[:, start:start + width].copy(), dtype=dtype)

    offset = 0
    window_x = extract(4, np.int32, offset); offset += 4
    window_y = extract(4, np.int32, offset); offset += 4

    coord_arrays = []
    if has_coords:
        coord_arrays = [
            extract(8, np.float64, offset),
            extract(8, np.float64, offset + 8),
            extract(8, np.float64, offset + 16),
        ]
        offset += 24

    tensor_arrays = []
    for _ in range(18):
        tensor_arrays.append(extract(8, np.float64, offset))
        offset += 8

    return (window_x, window_y, *coord_arrays, *tensor_arrays)


def read_text(file: str, delimiter: str):
    """Read a text strain result file.

    Expected formats are either 20 columns::

        window_x, window_y, def_grad_00..def_grad_22, eps_00..eps_22

    or 23 columns with ``x_mm``, ``y_mm`` and ``z_mm`` inserted after
    ``window_y``.
    """

    check_delimiter(file, delimiter)
    data = np.loadtxt(file, delimiter=delimiter, skiprows=1)
    if data.ndim == 1:
        data = data.reshape(1, -1)

    if data.shape[1] not in {15, 18}:
        raise ValueError(f"Text strain data must have exactly 15 or 18 columns. Number of cols = {data.shape[1]}")

    return (
        data[:, 0].astype(np.int32),
        data[:, 1].astype(np.int32),
        *[data[:, i] for i in range(2, data.shape[1])],
    )
