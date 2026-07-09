from __future__ import annotations

from pathlib import Path

import pytest

from pyvale.vfm.interpolate_fe_data_to_grid import (
    build_surface_geometry_from_gmsh,
    interpolate_fe_data_to_grid,
)


REPO_ROOT = Path(__file__).resolve().parents[2]
SINGLE_ELEMENT_DIR = REPO_ROOT / "dev" / "vfm" / "rob-data" / "single-element-plane-stress" / "fe-data"
PLATE_WITH_HOLE_DIR = REPO_ROOT / "dev" / "vfm" / "rob-data" / "plate-with-hole-hom-lin-hard"


def test_build_surface_geometry_from_single_element_mesh_returns_unit_area() -> None:
    geometry = build_surface_geometry_from_gmsh(SINGLE_ELEMENT_DIR / "single_element_square.msh")

    assert geometry.area == pytest.approx(1.0)


def test_interpolate_single_element_to_grid_returns_single_point() -> None:
    result = interpolate_fe_data_to_grid(
        SINGLE_ELEMENT_DIR / "element_data.csv",
        component_columns=("eps_xx", "eps_yy", "eps_xy"),
        mesh_path=SINGLE_ELEMENT_DIR / "single_element_square.msh",
        upsample_factor=2.0,
    )

    assert result.x_grid.shape == (1, 1)
    assert result.y_grid.shape == (1, 1)
    assert result.strain.shape == (1, 3, 1, 1)
    assert bool(result.specimen_mask[0, 0]) is True
    assert result.total_specimen_area == pytest.approx(1.0)


def test_interpolate_plate_with_hole_to_grid_preserves_hole_and_timesteps() -> None:
    result = interpolate_fe_data_to_grid(
        PLATE_WITH_HOLE_DIR / "element_data.csv",
        component_columns=("eps_xx", "eps_yy", "eps_xy"),
        mesh_path=PLATE_WITH_HOLE_DIR / "mesh2d_holeplate.msh",
        upsample_factor=1.0,
    )

    assert result.strain.shape[0] == 7
    assert result.strain.shape[1] == 3
    assert result.x_grid.shape == result.y_grid.shape
    assert result.specimen_mask.shape == result.x_grid.shape
    assert result.x_grid.shape[0] > 100
    assert result.x_grid.shape[1] > 100
    assert result.total_specimen_area is not None
    assert result.total_specimen_area > 0.0
    assert not result.specimen_mask.all()
    assert result.metadata["geometry"]["applied_scale_factor"] == pytest.approx(1000.0)

    x_axis = result.x_grid[0, :]
    y_axis = result.y_grid[:, 0]
    hole_col = int(abs(x_axis - 0.0).argmin())
    hole_row = int(abs(y_axis - 17.5).argmin())
    assert bool(result.specimen_mask[hole_row, hole_col]) is False
