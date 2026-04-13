from __future__ import annotations

from pathlib import Path

import numpy as np
from scipy.io import loadmat

from pyvale.vfm.project_definition import TestData


def parse_test_data_from_mat(mat_path: str | Path) -> TestData:
    """Load a MATLAB `testData.mat` file and rearrange it into toolkit form.

    The array reshaping follows the same convention used in `main_old.py`:
    strain is returned as `(timestep, component, y, x)` and the y-axis is
    flipped so row order increases downwards.
    """

    data = loadmat(
        mat_path,
        struct_as_record=False,
        squeeze_me=True,
        simplify_cells=True,
    )
    test_data = data["testData"]
    strain = test_data["strain"]

    x_raw = np.asarray(test_data["X"], dtype=np.float64)
    y_raw = np.asarray(test_data["Y"], dtype=np.float64)

    num_rows = int(x_raw.shape[0])
    num_cols = int(x_raw.shape[1])
    num_timesteps = int(strain["c11"].shape[-1] if strain["c11"].ndim > 1 else 1)

    strain_c11 = np.asarray(strain["c11"], dtype=np.float64).reshape(
        (num_rows, num_cols, num_timesteps),
        order="F",
    )
    strain_c22 = np.asarray(strain["c22"], dtype=np.float64).reshape(
        (num_rows, num_cols, num_timesteps),
        order="F",
    )
    strain_c12 = np.asarray(strain["c12"], dtype=np.float64).reshape(
        (num_rows, num_cols, num_timesteps),
        order="F",
    )

    strain_4d = np.stack(
        (
            np.transpose(strain_c11, (2, 0, 1)),
            np.transpose(strain_c22, (2, 0, 1)),
            np.transpose(strain_c12, (2, 0, 1)),
        ),
        axis=1,
    )
    strain_4d = np.flip(strain_4d, axis=2)

    x_values = np.nanmean(x_raw, axis=0, keepdims=True)
    x = np.tile(x_values, (x_raw.shape[0], 1))

    y_values = np.nanmean(y_raw, axis=1, keepdims=True)
    y = np.tile(y_values, (1, y_raw.shape[1]))
    y = np.flip(y, axis=0)

    specimen_mask = ~np.isnan(x_raw)
    specimen_mask = np.flip(specimen_mask, axis=0)

    area = np.asarray(test_data["area"], dtype=np.float64).reshape(
        (num_rows, num_cols),
        order="F",
    )
    area = np.flip(area, axis=0)

    force = np.asarray(test_data["FGlob"], dtype=np.float64)
    time_data = test_data["time"]
    if isinstance(time_data, dict):
        time = np.asarray(time_data["time"], dtype=np.float64)
    else:
        time = np.asarray(time_data, dtype=np.float64)

    return TestData(
        x=x,
        y=y,
        specimen_mask=specimen_mask,
        area=area,
        strain=strain_4d,
        force=force,
        time=time,
        source_path=Path(mat_path),
    )


def save_parsed_test_data(
    test_data: TestData,
    output_path: str | Path,
) -> Path:
    """Save parsed test data to a Python-side `.npz` archive."""

    target_path = Path(output_path)
    if target_path.suffix != ".npz":
        target_path = target_path.with_suffix(".npz")

    np.savez_compressed(
        target_path,
        x=test_data.x,
        y=test_data.y,
        specimen_mask=test_data.specimen_mask,
        area=test_data.area,
        strain=test_data.strain,
        force=test_data.force,
        time=test_data.time,
        source_path=(
            "" if test_data.source_path is None else str(test_data.source_path)
        ),
    )
    return target_path


def load_parsed_test_data(npz_path: str | Path) -> TestData:
    """Load a `.npz` archive previously written by `save_parsed_test_data`."""

    with np.load(npz_path) as saved_data:
        source_path_text = str(saved_data["source_path"])
        source_path = Path(source_path_text) if source_path_text else None
        return TestData(
            x=np.asarray(saved_data["x"], dtype=np.float64),
            y=np.asarray(saved_data["y"], dtype=np.float64),
            specimen_mask=np.asarray(saved_data["specimen_mask"], dtype=bool),
            area=np.asarray(saved_data["area"], dtype=np.float64),
            strain=np.asarray(saved_data["strain"], dtype=np.float64),
            force=np.asarray(saved_data["force"], dtype=np.float64),
            time=np.asarray(saved_data["time"], dtype=np.float64),
            source_path=source_path,
        )


def convert_mat_to_py_data(
    mat_path: str | Path,
    output_path: str | Path | None = None,
) -> Path:
    """Convert a MATLAB `testData.mat` file into a saved Python `.npz` file."""

    mat_path = Path(mat_path)
    test_data = parse_test_data_from_mat(mat_path)
    if output_path is None:
        output_path = mat_path.with_suffix(".npz")
    return save_parsed_test_data(test_data, output_path)

# Example usage:
# convert_mat_to_py_data("path/to/testData.mat", "path/to/parsed_test_data.npz")
# in terminal: 
#  PYTHONPATH=src python -c "from pyvale.vfm.mat_to_py_data_parser import convert_mat_to_py_data; convert_mat_to_py_data('/home/robh/1_Projects/vfmap-numerical-paper/data/notchedButtWeld_bilin_lin360420S_hom3700H_imDef_1.5/5-testData/testData.mat', '/home/robh/1_Projects/vfmap-numerical-paper/data/notchedButtWeld_bilin_lin360420S_hom3700H_imDef_1.5/5-testData/test_data.npz')"


