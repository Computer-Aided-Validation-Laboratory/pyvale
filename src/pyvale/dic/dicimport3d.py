# ================================================================================
# pyvale: the python validation engine
# License: MIT
# Copyright (C) 2025 The Computer Aided Validation Team
# ================================================================================

import numpy as np
import glob
from pathlib import Path
from typing import Literal

# Pyvale modules
from pyvale.dic.dicresults import Results, StereoResults
from pyvale.dic.dicimport2d import to_grid, check_delimiter


def import_3d(data: str | Path | list[Path],
              delimiter: str,
              binary: bool = False,
              layout: Literal["column", "matrix"] = "matrix") -> Results:
    """
    Import stereo DIC result data from human readable text or binary files.

    Parameters
    ----------
    data : str or pathlib.Path or list[pathlib.Path]
        Path pattern to the data files (can include wildcards).

    layout : str, optional
        Format of the output data layout:
        "column" or "matrix".

    binary : bool, optional
        If True, expects binary files.

    delimiter : str, optional
        Delimiter used in text files.

    Returns
    -------
    Results
        Results object containing both 2D DIC and stereo DIC data.
    """

    print("")
    print("Attempting Stereo DIC Data import...")
    print("")

    # convert to str
    if isinstance(data, Path):
        data = str(data)

    if isinstance(data, list):
        files = list(map(str, data))
    else:
        files = sorted(glob.glob(data))
        if not files:
            raise FileNotFoundError(f"No results found in: {data}")

    print(f"Found {len(files)} files containing stereo DIC results:")
    for file in files:
        print(f"  - {file}")
    print("")

    read_data = read_binary_3d if binary else read_text_3d

    ss_x_ref, ss_y_ref, *fields = read_data(
        files[0],
        delimiter=delimiter,
    )

    frames = [list(fields)]

    for file in files[1:]:

        ss_x, ss_y, *f = read_data(
            file,
            delimiter=delimiter,
        )

        if not (
            np.array_equal(ss_x_ref, ss_x)
            and np.array_equal(ss_y_ref, ss_y)
        ):
            raise ValueError(
                "Mismatch in coordinates across frames."
            )

        frames.append(f)

    arrays = [
        np.stack([frame[i] for frame in frames])
        for i in range(len(fields))
    ]

    if layout == "matrix":

        x_unique = np.unique(ss_x_ref)
        y_unique = np.unique(ss_y_ref)

        X, Y = np.meshgrid(
            x_unique,
            y_unique,
        )

        shape = (
            len(files),
            len(y_unique),
            len(x_unique),
        )

        arrays = [
            to_grid(
                a,
                shape,
                ss_x_ref,
                ss_y_ref,
                x_unique,
                y_unique,
            )
            for a in arrays
        ]

        ss_x_out = X
        ss_y_out = Y

    else:

        ss_x_out = ss_x_ref
        ss_y_out = ss_y_ref

    return Results(
        ss_x=ss_x_out,
        ss_y=ss_y_out,

        u=arrays[0],
        v=arrays[1],
        mag=arrays[2],

        converged=arrays[3],
        cost=arrays[4],
        ftol=arrays[5],
        xtol=arrays[6],
        niter=arrays[7],

        stereo=StereoResults(
            u_px=arrays[8],
            v_px=arrays[9],
            mag_px=arrays[10],

            u_mm=arrays[11],
            v_mm=arrays[12],
            w_mm=arrays[13],

            x_mm=arrays[14],
            y_mm=arrays[15],
            z_mm=arrays[16],

            converged=arrays[17],
            cost=arrays[18],
            ftol=arrays[19],
            xtol=arrays[20],
            niter=arrays[21],
        ),

        filenames=files,
    )


def read_text_3d(
    file: str,
    delimiter: str,
):
    """
    Read a human-readable stereo DIC result file.

    Expected columns
    ----------------
    subset_x
    subset_y
    disp_u
    disp_v
    disp_mag
    converged
    cost_zncc
    ftol
    xtol
    num_iter
    stereo_disp_u_px
    stereo_disp_v_px
    stereo_disp_mag_px
    stereo_disp_u_mm
    stereo_disp_v_mm
    stereo_disp_w_mm
    stereo_x_mm
    stereo_y_mm
    stereo_z_mm
    stereo_converged
    stereo_cost_zncc
    stereo_ftol
    stereo_xtol
    stereo_num_iter
    """

    check_delimiter(file, delimiter)
    data = np.loadtxt(
        file,
        delimiter=delimiter,
        skiprows=1,
    )

    if data.shape[1] != 24:
        raise ValueError(
            "Input stereo DIC data must have exactly 24 columns."
        )

    return (
        data[:, 0].astype(np.int32),   # subset_x
        data[:, 1].astype(np.int32),   # subset_y

        data[:, 2],                    # disp_u
        data[:, 3],                    # disp_v
        data[:, 4],                    # disp_mag

        data[:, 5].astype(np.bool_),   # converged
        data[:, 6],                    # cost
        data[:, 7],                    # ftol
        data[:, 8],                    # xtol
        data[:, 9].astype(np.int32),   # num_iter

        data[:, 10],                   # stereo_disp_u_px
        data[:, 11],                   # stereo_disp_v_px
        data[:, 12],                   # stereo_disp_mag_px

        data[:, 13],                   # stereo_disp_u_mm
        data[:, 14],                   # stereo_disp_v_mm
        data[:, 15],                   # stereo_disp_w_mm

        data[:, 16],                   # stereo_x_mm
        data[:, 17],                   # stereo_y_mm
        data[:, 18],                   # stereo_z_mm

        data[:, 19].astype(np.bool_),  # stereo_converged
        data[:, 20],                   # stereo_cost
        data[:, 21],                   # stereo_ftol
        data[:, 22],                   # stereo_xtol
        data[:, 23].astype(np.int32),  # stereo_num_iter
    )

def read_binary_3d(file: str, delimiter: str):
    """
    Read a binary stereo DIC result file and extract all fields.

    Must match ResultArrays::write_to_disk_stereo exactly.
    """

    with open(file, "rb") as f:
        raw = f.read()

    row_size = (
        4 * 4 +   # int32: ss_x, ss_y, stereo uses same grid (2 ints total here actually but packed in 4 ints total due to second block)
        2 * 1 +   # uint8: conv, stereo.conv
        22 * 8    # all doubles
    )

    file_size = len(raw)

    if file_size % row_size != 0:
        raise ValueError(
            f"Binary file has incomplete rows: {file}. "
            f"Expected row size: {row_size}, "
            f"Actual size: {file_size} bytes."
        )

    rows = file_size // row_size

    arr = np.frombuffer(raw, dtype=np.uint8).reshape(rows, row_size)

    def extract(col, dtype, start):
        return np.frombuffer(
            arr[:, start:start + col].copy(),
            dtype=dtype,
        )

    offset = 0

    # -----------------------
    # 2D base fields
    # -----------------------
    ss_x = extract(4, np.int32, offset); offset += 4
    ss_y = extract(4, np.int32, offset); offset += 4

    u = extract(8, np.float64, offset); offset += 8
    v = extract(8, np.float64, offset); offset += 8
    mag = extract(8, np.float64, offset); offset += 8

    conv = extract(1, np.uint8, offset).astype(bool); offset += 1

    cost = extract(8, np.float64, offset); offset += 8
    ftol = extract(8, np.float64, offset); offset += 8
    xtol = extract(8, np.float64, offset); offset += 8

    niter = extract(4, np.int32, offset); offset += 4

    # -----------------------
    # stereo pixel fields
    # -----------------------
    stereo_u_px = extract(8, np.float64, offset); offset += 8
    stereo_v_px = extract(8, np.float64, offset); offset += 8
    stereo_mag_px = extract(8, np.float64, offset); offset += 8

    # -----------------------
    # stereo world fields
    # -----------------------
    stereo_u_mm = extract(8, np.float64, offset); offset += 8
    stereo_v_mm = extract(8, np.float64, offset); offset += 8
    stereo_w_mm = extract(8, np.float64, offset); offset += 8

    stereo_x_mm = extract(8, np.float64, offset); offset += 8
    stereo_y_mm = extract(8, np.float64, offset); offset += 8
    stereo_z_mm = extract(8, np.float64, offset); offset += 8

    # -----------------------
    # stereo convergence
    # -----------------------
    stereo_conv = extract(1, np.uint8, offset).astype(bool); offset += 1

    stereo_cost = extract(8, np.float64, offset); offset += 8
    stereo_ftol = extract(8, np.float64, offset); offset += 8
    stereo_xtol = extract(8, np.float64, offset); offset += 8

    stereo_niter = extract(4, np.int32, offset); offset += 4

    return (
        ss_x,
        ss_y,

        u,
        v,
        mag,
        conv,
        cost,
        ftol,
        xtol,
        niter,

        stereo_u_px,
        stereo_v_px,
        stereo_mag_px,

        stereo_u_mm,
        stereo_v_mm,
        stereo_w_mm,

        stereo_x_mm,
        stereo_y_mm,
        stereo_z_mm,

        stereo_conv,
        stereo_cost,
        stereo_ftol,
        stereo_xtol,
        stereo_niter,
    )