import json
from pathlib import Path

import numpy as np
import pytest

from pyvale.vfm.inputdataprocessor import (
    BOUNDARY_CONDITIONS,
    InputDataConfig,
    LoadedFieldData,
    _apply_coordinate_convention_to_axis_grid,
    _apply_coordinate_convention_to_force,
    _apply_coordinate_convention_to_strain,
    _determine_coordinate_convention_transform,
    _load_field_data,
    _load_spatial_selection_info,
    _maybe_flip_force_sign,
)
from pyvale.vfm.roi import RoiDefinition, RoiShape, sample_roi_definition_at_coordinates


def _make_config(tmp_path: Path, **overrides) -> InputDataConfig:
    defaults = dict(
        input_folder=tmp_path,
        output_folder=tmp_path / "output",
        x_coordinates_input_file="x.csv",
        y_coordinates_input_file="y.csv",
        strain_input_file="strain.npy",
        force_input_file="force.csv",
        time_input_file="time.csv",
    )
    defaults.update(overrides)
    return InputDataConfig(**defaults)


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

    config = _make_config(
        tmp_path,
        strain_input_file="strain.npy",
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


def test_polygon_roi_sampling_keeps_full_rectangle() -> None:
    x = np.array([[0.0, 1.0], [0.0, 1.0]], dtype=np.float64)
    y = np.array([[0.0, 0.0], [1.0, 1.0]], dtype=np.float64)
    roi_definition = RoiDefinition(
        shapes=(
            RoiShape(
                shape_type="polygon",
                index=0,
                is_cutting=False,
                vertices=((0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)),
            ),
        )
    )

    sampled_mask = sample_roi_definition_at_coordinates(roi_definition, x, y)

    np.testing.assert_array_equal(sampled_mask, np.ones((2, 2), dtype=bool))


def test_maybe_flip_force_sign_applies_user_requested_sign_change(monkeypatch: pytest.MonkeyPatch) -> None:
    force = np.array([1.0, -2.0, 3.0], dtype=np.float64)
    monkeypatch.setattr("builtins.input", lambda _: "y")

    flipped_force, applied = _maybe_flip_force_sign(force)

    assert applied is True
    np.testing.assert_array_equal(flipped_force, -force)


def test_load_field_data_gridded_dic_path(tmp_path: Path) -> None:
    """The non-FE branch loads gridded strain/coords into a LoadedFieldData."""
    strain = np.arange(2 * 3 * 2 * 3, dtype=np.float64).reshape(2, 3, 2, 3)
    np.save(tmp_path / "strain.npy", strain)
    (tmp_path / "x.csv").write_text("0,1,2\n0,1,2\n", encoding="utf-8")
    (tmp_path / "y.csv").write_text("0,0,0\n1,1,1\n", encoding="utf-8")

    config = _make_config(tmp_path)

    data = _load_field_data(config)

    assert isinstance(data, LoadedFieldData)
    # No FE interpolation and no pre-computed mask on the DIC path.
    assert data.fe_interpolation is None
    assert data.specimen_mask is None
    assert data.roi_summary is None
    assert data.total_area_override is None
    assert data.strain.shape == (2, 3, 2, 3)
    assert data.x_raw.shape == (2, 3)
    # ROI-alignment coords fall back to the main coords when no pixel grids given.
    np.testing.assert_array_equal(data.roi_alignment_x, data.x_raw)
    np.testing.assert_array_equal(
        data.original_coordinate_valid_mask, np.ones((2, 3), dtype=bool)
    )
