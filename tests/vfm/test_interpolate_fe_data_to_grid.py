from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
import numpy.testing as npt
import pytest

from pyvale.vfm.inputdatafeinterp import (
    build_surface_geometry_from_gmsh,
    interpolate_fe_data_to_grid,
)


COMPONENT_COLUMNS = ("eps_xx", "eps_yy", "eps_xy")


def test_build_surface_geometry_from_generated_quad_mesh(
    tmp_path: Path,
) -> None:
    mesh_path = _write_quad_mesh(tmp_path / "unit_square.msh")

    geometry = build_surface_geometry_from_gmsh(mesh_path)

    assert geometry.area == pytest.approx(1.0)
    assert geometry.bounds == pytest.approx((0.0, 0.0, 1.0, 1.0))


def test_interpolate_generated_single_element_table_uses_direct_copy(
    tmp_path: Path,
) -> None:
    mesh_path = _write_quad_mesh(tmp_path / "unit_square.msh")
    table_path = _write_element_table(
        tmp_path / "single_element.csv",
        points=((1, 0.5, 0.5),),
        times=(0.0, 1.0),
    )

    result = interpolate_fe_data_to_grid(
        table_path,
        component_columns=COMPONENT_COLUMNS,
        mesh_path=mesh_path,
        upsample_factor=2.0,
    )

    assert result.x_grid.shape == (1, 1)
    assert result.y_grid.shape == (1, 1)
    assert result.strain.shape == (2, 3, 1, 1)
    assert bool(result.specimen_mask[0, 0]) is True
    assert result.total_specimen_area == pytest.approx(1.0)
    assert result.metadata["interpolation_method"] == "direct-single-point-copy"
    npt.assert_allclose(result.strain[:, 0, 0, 0], [0.5, 1.5])


def test_interpolate_generated_ring_mesh_preserves_hole_and_timesteps(
    tmp_path: Path,
) -> None:
    mesh_path, points = _write_ring_mesh_in_metres(tmp_path / "ring.msh")
    table_path = _write_element_table(
        tmp_path / "ring_elements.csv",
        points=points,
        times=(0.0, 1.0),
    )

    result = interpolate_fe_data_to_grid(
        table_path,
        component_columns=COMPONENT_COLUMNS,
        mesh_path=mesh_path,
        target_spacing=0.5,
    )

    assert result.strain.shape == (2, 3, 7, 7)
    assert result.specimen_mask.shape == (7, 7)
    assert result.total_specimen_area == pytest.approx(8.0)
    assert result.metadata["geometry"]["applied_scale_factor"] == pytest.approx(
        1000.0
    )

    x_axis = result.x_grid[0]
    y_axis = result.y_grid[:, 0]
    hole_column = int(np.abs(x_axis - 1.5).argmin())
    hole_row = int(np.abs(y_axis - 1.5).argmin())
    assert bool(result.specimen_mask[hole_row, hole_column]) is False
    assert np.all(np.isnan(result.strain[:, :, hole_row, hole_column]))
    assert np.all(np.isfinite(result.strain[:, :, result.specimen_mask]))


def _write_quad_mesh(path: Path) -> Path:
    path.write_text(
        "\n".join(
            (
                "$MeshFormat",
                "2.2 0 8",
                "$EndMeshFormat",
                "$Nodes",
                "4",
                "1 0 0 0",
                "2 1 0 0",
                "3 1 1 0",
                "4 0 1 0",
                "$EndNodes",
                "$Elements",
                "1",
                "1 3 0 1 2 3 4",
                "$EndElements",
                "",
            )
        ),
        encoding="utf-8",
    )
    return path


def _write_ring_mesh_in_metres(
    path: Path,
) -> tuple[Path, tuple[tuple[int, float, float], ...]]:
    node_lines: list[str] = []
    for row in range(4):
        for column in range(4):
            node_id = row * 4 + column + 1
            node_lines.append(
                f"{node_id} {column * 0.001:.6f} {row * 0.001:.6f} 0"
            )

    element_lines: list[str] = []
    points: list[tuple[int, float, float]] = []
    element_id = 1
    for row in range(3):
        for column in range(3):
            if row == 1 and column == 1:
                continue
            lower_left = row * 4 + column + 1
            lower_right = lower_left + 1
            upper_left = lower_left + 4
            upper_right = upper_left + 1
            element_lines.append(
                f"{element_id} 3 0 {lower_left} {lower_right} "
                f"{upper_right} {upper_left}"
            )
            points.append((element_id, column + 0.5, row + 0.5))
            element_id += 1

    path.write_text(
        "\n".join(
            (
                "$MeshFormat",
                "2.2 0 8",
                "$EndMeshFormat",
                "$Nodes",
                str(len(node_lines)),
                *node_lines,
                "$EndNodes",
                "$Elements",
                str(len(element_lines)),
                *element_lines,
                "$EndElements",
                "",
            )
        ),
        encoding="utf-8",
    )
    return path, tuple(points)


def _write_element_table(
    path: Path,
    *,
    points: tuple[tuple[int, float, float], ...],
    times: tuple[float, ...],
) -> Path:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=("element_id", "time", "x", "y", *COMPONENT_COLUMNS),
        )
        writer.writeheader()
        for element_id, x_coordinate, y_coordinate in points:
            for time in times:
                writer.writerow(
                    {
                        "element_id": element_id,
                        "time": time,
                        "x": x_coordinate,
                        "y": y_coordinate,
                        "eps_xx": x_coordinate + time,
                        "eps_yy": y_coordinate - time,
                        "eps_xy": x_coordinate + y_coordinate + 0.5 * time,
                    }
                )
    return path
