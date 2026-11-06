# ============================================================================
# pyvale: the python validation engine
# License: MIT
# Copyright (C) 2026 Sceptical Rabbit (Lloyd Fletcher)
# ============================================================================
"""Tests for the consolidated packaged-data namespace."""

import pyvale.data as dataset
import riley


def test_dataset_accessors_resolve_consolidated_resources() -> None:
    """Public accessors resolve assets below the single data package."""
    paths = (
        dataset.mechanical_2d_path(),
        dataset.sim_case_input_file_path(17),
        dataset.sim_case_gmsh_file_path(17),
        dataset.dic_plate_with_hole_cam0_ref(),
        dataset.dic_pattern_5mpx_path(),
        dataset.cal_target(),
    )
    assert all(path is not None and path.exists() for path in paths)


def test_riley_data_accessors_resolve_packaged_resources() -> None:
    """Riley-owned example assets resolve from the Riley package."""
    paths = (
        riley.data.speckle_texture_path(),
        riley.data.cal_target_texture_path(),
        riley.data.sphere200_case_path(),
        riley.data.platehole_csv_case_path(),
        riley.data.platehole_exodus_path(),
        riley.data.stereocal_case_path(),
        riley.data.rabbit_case_path("riley", "tri3"),
    )
    assert all(path.exists() for path in paths)
