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
from pyvale.dic.dicimport2d import check_delimiter, to_grid


def import_2d(data: str | Path | list[Path],
              binary: bool = False,
              layout: Literal["column", "matrix"] = "matrix",
              delimiter: str = ",") -> StrainResults:
    """Import 2D strain result data from text or binary files.

    The importer expects the current format containing the two trailing
    in-plane principal-strain columns, ``eps1`` and ``eps2``. Files also
    contain ``x_mm``, ``y_mm`` and ``z_mm`` after the window pixel coordinates.
    """

    return _import(data, binary=binary, layout=layout, delimiter=delimiter, require_coords=False)


def import_3d(data: str | Path | list[Path],
              binary: bool = False,
              layout: Literal["column", "matrix"] = "matrix",
              delimiter: str = ",") -> StrainResults:
    """Import 3D-aware strain result data from text or binary files.

    Returned ``StrainResults`` include ``x_mm``, ``y_mm`` and ``z_mm`` arrays for
    the strain-window centre in the cam0 coordinate system.
    """

    return _import(data, binary=binary, layout=layout, delimiter=delimiter, require_coords=True)


def _import(data: str | Path | list[Path],
            binary: bool,
            layout: Literal["column", "matrix"],
            delimiter: str,
            require_coords: bool) -> StrainResults:
    if layout not in {"column", "matrix"}:
        raise ValueError("layout must be 'column' or 'matrix'.")

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

    first = read_file(files[0])
    window_x_ref, window_y_ref = first[:2]
    frames = [first[2:]]

    for file in files[1:]:
        current = read_file(file)
        window_x, window_y = current[:2]
        if not (np.array_equal(window_x_ref, window_x) and np.array_equal(window_y_ref, window_y)):
            raise ValueError("Mismatch in coordinates across frames.")
        frames.append(current[2:])

    arrays = [np.stack([frame[i] for frame in frames]) for i in range(len(frames[0]))]

    if len(arrays) != 15:
        raise ValueError( "Strain data must be in 17-column format.")

    x_mm, y_mm, z_mm = arrays[:3]
    tensor_arrays = arrays[3:]
    principal_arrays = tensor_arrays[10:12]
    tensor_arrays = tensor_arrays[:10]

    if layout == "matrix":
        x_unique = np.unique(window_x_ref)
        y_unique = np.unique(window_y_ref)
        window_x_out, window_y_out = np.meshgrid(x_unique, y_unique)
        shape = (len(files), len(y_unique), len(x_unique))
        x_indices = np.searchsorted(x_unique, window_x_ref)
        y_indices = np.searchsorted(y_unique, window_y_ref)

        tensor_arrays = [
            to_grid(a, shape, x_indices, y_indices)
            for a in tensor_arrays
        ]

        principal_arrays = [
            to_grid(a, shape, x_indices, y_indices)
            for a in principal_arrays
        ]

        x_mm = to_grid(x_mm, shape, x_indices, y_indices)
        y_mm = to_grid(y_mm, shape, x_indices, y_indices)
        z_mm = to_grid(z_mm, shape, x_indices, y_indices)
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
