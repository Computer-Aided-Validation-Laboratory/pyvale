from pathlib import Path

import numpy as np
import pytest

import pyvale.dataio as io
from pyvale.dataio.exceptions import ExpLoadErr


def test_column_loader_aligns_irregular_points_by_id(tmp_path: Path) -> None:
    frame_zero = tmp_path / "frame_0.csv"
    frame_one = tmp_path / "frame_1.csv"
    frame_zero.write_text("id,x,y,u\na,1,2,0\nb,4,1,2\nc,6,8,3\n")
    frame_one.write_text("id,x,y,u\nc,6,8,13\na,1,2,10\nb,4,1,NaN\n")

    loaded = io.CameraLoaderColumn(
        [frame_zero, frame_one],
        coord_cols=(1, 2),
        component_cols={"u": 3},
        point_id_col=0,
        times=np.array([0.0, 1.0]),
    ).load_data()

    assert loaded.component_labels == ("u",)
    assert loaded.grid_shape is None
    assert loaded.sens_label_to_ind == {"a": 0, "b": 1, "c": 2}
    np.testing.assert_array_equal(loaded.coords, [[1, 2, 0], [4, 1, 0], [6, 8, 0]])
    np.testing.assert_array_equal(loaded.fields, [[[0, 10]], [[2, np.nan]], [[3, 13]]])


@pytest.mark.parametrize(
    ("rows", "message"),
    [
        ("b,3,4,5\nb,1,2,5\n", "Duplicate point IDs"),
        ("a,1,2,5\nc,3,4,5\n", "Point IDs or row count differ"),
        ("a,1,9,5\nb,3,4,5\n", "changed reference coordinates"),
    ],
)
def test_column_loader_rejects_inconsistent_frames(
    tmp_path: Path, rows: str, message: str
) -> None:
    frame_zero = tmp_path / "frame_0.csv"
    frame_one = tmp_path / "frame_1.csv"
    frame_zero.write_text("id,x,y,u\na,1,2,0\nb,3,4,1\n")
    frame_one.write_text("id,x,y,u\n" + rows)

    loader = io.CameraLoaderColumn(
        [frame_zero, frame_one],
        coord_cols=(1, 2),
        component_cols={"u": 3},
        point_id_col=0,
    )

    with pytest.raises(ExpLoadErr, match=message):
        loader.load_data()


def test_matrix_loader_preserves_grid_and_singleton_axes(tmp_path: Path) -> None:
    x_path = tmp_path / "x.csv"
    y_path = tmp_path / "y.csv"
    field_path = tmp_path / "u.csv"
    x_path.write_text("1,2\n1,2\n")
    y_path.write_text("3,3\n4,4\n")
    field_path.write_text("1,2\n3,NaN\n")
    loader = io.CameraLoaderMatrix([{"u": field_path}], coord_files=(x_path, y_path))

    loaded = loader.load_data()

    assert loaded.grid_shape == (2, 2)
    assert loaded.fields.shape == (4, 1, 1)
    np.testing.assert_array_equal(
        loaded.coords, [[1, 3, 0], [2, 3, 0], [1, 4, 0], [2, 4, 0]]
    )
    np.testing.assert_array_equal(loaded.fields[:, 0, 0], [1, 2, 3, np.nan])

    field_path.write_text("1,2\n")
    with pytest.raises(ExpLoadErr, match="expected"):
        loader.load_data()
