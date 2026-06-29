from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from pyvale.vfm.prepareinputdata import (
    BOUNDARY_CONDITIONS,
    PreparationConfig,
    _apply_coordinate_convention_to_axis_grid,
    _apply_coordinate_convention_to_force,
    _apply_coordinate_convention_to_strain,
    _determine_coordinate_convention_transform,
    _load_spatial_selection_info,
)


def test_determine_coordinate_convention_transform_detects_both_axis_reflections() -> None:
    x = np.array([[3.0, 2.0], [3.0, 2.0]], dtype=np.float64)
    y = np.array([[4.0, 4.0], [1.0, 1.0]], dtype=np.float64)

    transform = _determine_coordinate_convention_transform(x, y)

    assert transform.reflect_x is True
    assert transform.reflect_y is True
    assert transform.applied is True


def test_apply_coordinate_convention_to_strain_negates_shear_for_single_axis_reflection() -> None:
    strain = np.ones((2, 3, 2, 4), dtype=np.float64)
    strain[:, 2, :, :] = 7.5
    transform = _determine_coordinate_convention_transform(
        np.array([[3.0, 2.0, 1.0, 0.0], [3.0, 2.0, 1.0, 0.0]], dtype=np.float64),
        np.array([[1.0, 1.0, 1.0, 1.0], [2.0, 2.0, 2.0, 2.0]], dtype=np.float64),
    )

    transformed_strain = _apply_coordinate_convention_to_strain(
        strain,
        transform,
        component_names=("exx", "eyy", "exy"),
    )

    np.testing.assert_array_equal(transformed_strain[:, 0, :, :], strain[:, 0, :, :])
    np.testing.assert_array_equal(transformed_strain[:, 1, :, :], strain[:, 1, :, :])
    np.testing.assert_array_equal(transformed_strain[:, 2, :, :], -strain[:, 2, :, :])


def test_load_spatial_selection_info_reads_metadata_and_indices_file(tmp_path: Path) -> None:
    metadata_path = tmp_path / "metadata.json"
    metadata_path.write_text(
        json.dumps(
            {
                "selection": {
                    "selected_spatial_row_indices": [10, 11, 12],
                    "selected_spatial_column_indices": [20, 21],
                }
            }
        ),
        encoding="utf-8",
    )
    indices_path = tmp_path / "selected_spatial_indices.txt"
    indices_path.write_text(
        "selected_spatial_row_indices = [10, 11, 12]\n"
        "selected_spatial_column_indices = [20, 21]\n",
        encoding="utf-8",
    )

    config = PreparationConfig(
        input_folder=tmp_path,
        output_folder=tmp_path / "output",
        x_coordinates_input_file="x.csv",
        y_coordinates_input_file="y.csv",
        strain_input_file="strain.npy",
        force_input_file="force.csv",
        time_input_file="time.csv",
        metadata_file=metadata_path.name,
        selected_spatial_indices_file=indices_path.name,
    )

    info = _load_spatial_selection_info(config, target_shape=(3, 2))

    assert info is not None
    assert info["row_count"] == 3
    assert info["column_count"] == 2
    assert info["row_range"] == [10, 12]
    assert info["column_range"] == [20, 21]
    assert info["matches_prepared_grid_shape"] is True


def test_apply_coordinate_convention_to_axis_grid_reflects_values_in_place() -> None:
    x = np.array([[3.0, 2.0], [3.0, 2.0]], dtype=np.float64)

    transformed_x = _apply_coordinate_convention_to_axis_grid(x, reflect_axis=True)

    np.testing.assert_array_equal(transformed_x, np.array([[2.0, 3.0], [2.0, 3.0]], dtype=np.float64))


def test_apply_coordinate_convention_to_force_negates_reflected_component() -> None:
    force = np.array([1.0, -2.0, 3.0], dtype=np.float64)
    transform = _determine_coordinate_convention_transform(
        np.array([[1.0, 2.0], [1.0, 2.0]], dtype=np.float64),
        np.array([[3.0, 3.0], [1.0, 1.0]], dtype=np.float64),
    )

    transformed_force = _apply_coordinate_convention_to_force(
        force,
        transform,
        boundary_conditions=BOUNDARY_CONDITIONS,
    )

    np.testing.assert_array_equal(transformed_force, -force)
