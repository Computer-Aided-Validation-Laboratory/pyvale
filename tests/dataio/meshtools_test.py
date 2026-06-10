# ==============================================================================
# pyvale: the python validation engine
# License: MIT
# Copyright (C) 2025 The Computer Aided Validation Team
# ==============================================================================

import numpy as np
import pytest

import pyvale.dataio as io


def _quad_coords() -> np.ndarray:
    return np.array(
        (
            (0.0, 0.0, 0.0),
            (1.0, 0.0, 0.0),
            (1.0, 1.0, 0.0),
            (0.0, 1.0, 0.0),
        ),
        dtype=np.float64,
    )


def _tet_coords(offset: float = 0.0) -> np.ndarray:
    return np.array(
        (
            (offset + 0.0, 0.0, 0.0),
            (offset + 1.0, 0.0, 0.0),
            (offset + 0.0, 1.0, 0.0),
            (offset + 0.0, 0.0, 1.0),
        ),
        dtype=np.float64,
    )


def test_check_mesh_convention_passes_for_canonical_quad() -> None:
    mesh = io.SimData(
        num_spat_dims=2,
        coords=_quad_coords(),
        connect={"connect1": np.array(((0, 1, 2, 3),), dtype=np.int64)},
    )

    check = io.check_mesh_convention(mesh)

    assert check.is_valid
    assert check.failed_checks == tuple()
    assert check.connectivity_failures["connect1"] == tuple()


def test_enforce_mesh_convention_corrects_legacy_connectivity() -> None:
    mesh = io.SimData(
        num_spat_dims=2,
        coords=_quad_coords(),
        connect={"connect1": np.array(((1,), (2,), (3,), (4,)), dtype=np.int64)},
    )

    mesh_out = io.enforce_mesh_convention(mesh)

    assert mesh_out is not mesh
    assert mesh_out.connect is not None
    assert mesh_out.connect["connect1"].shape == (1, 4)
    assert np.array_equal(mesh_out.connect["connect1"], np.array(((0, 1, 2, 3),)))


def test_check_mesh_convention_reports_failed_checks() -> None:
    mesh = io.SimData(
        num_spat_dims=2,
        coords=_quad_coords(),
        connect={"connect1": np.array(((1,), (4,), (3,), (2,)), dtype=np.int64)},
    )

    check = io.check_mesh_convention(mesh)

    assert not check.is_valid
    assert "zero_based_indexing" in check.failed_checks
    assert "row_major_connectivity" in check.failed_checks
    assert "ccw_winding" in check.failed_checks


def test_enforce_mesh_convention_raises_for_invalid_indices() -> None:
    mesh = io.SimData(
        num_spat_dims=2,
        coords=_quad_coords(),
        connect={"connect1": np.array(((0, 1, 2, 10),), dtype=np.int64)},
    )

    with pytest.raises(ValueError, match="invalid|outside"):
        io.enforce_mesh_convention(mesh)


def test_check_and_enforce_winding_for_quads() -> None:
    mesh = io.SimData(
        num_spat_dims=2,
        coords=_quad_coords(),
        connect={"connect1": np.array(((0, 3, 2, 1),), dtype=np.int64)},
    )

    assert io.check_cw_winding(mesh)
    assert not io.check_ccw_winding(mesh)

    mesh_out = io.enforce_ccw_winding(mesh)

    assert io.check_ccw_winding(mesh_out)
    assert np.array_equal(mesh_out.connect["connect1"], np.array(((0, 1, 2, 3),)))


def test_enforce_mesh_convention_fixes_tet_handedness() -> None:
    mesh = io.SimData(
        num_spat_dims=3,
        coords=_tet_coords(),
        connect={"connect1": np.array(((0, 2, 1, 3),), dtype=np.int64)},
    )

    check = io.check_mesh_convention(mesh)
    assert not check.is_right_handed

    mesh_out = io.enforce_mesh_convention(mesh)

    assert io.check_mesh_convention(mesh_out).is_valid
    assert np.array_equal(mesh_out.connect["connect1"], np.array(((0, 1, 2, 3),)))


def test_extract_surf_mesh_handles_multiple_connectivity_tables() -> None:
    coords = np.vstack((_tet_coords(0.0), _tet_coords(2.0)))
    mesh = io.SimData(
        num_spat_dims=3,
        coords=coords,
        connect={
            "connect1": np.array(((0, 1, 2, 3),), dtype=np.int64),
            "connect2": np.array(((4, 5, 6, 7),), dtype=np.int64),
        },
    )

    surf_mesh = io.extract_surf_mesh(mesh)

    assert surf_mesh.connect is not None
    assert tuple(surf_mesh.connect.keys()) == ("connect1", "connect2")
    assert surf_mesh.connect["connect1"].shape == (4, 3)
    assert surf_mesh.connect["connect2"].shape == (4, 3)
    assert np.all(surf_mesh.connect["connect1"] < surf_mesh.coords.shape[0])
    assert np.all(surf_mesh.connect["connect2"] < surf_mesh.coords.shape[0])


def test_extract_surf_mesh_preserves_legacy_style_when_requested() -> None:
    mesh = io.SimData(
        num_spat_dims=3,
        coords=_tet_coords(),
        connect={"connect1": np.array(((1,), (2,), (3,), (4,)), dtype=np.int64)},
    )

    surf_mesh = io.extract_surf_mesh(mesh, enforce_convention=False)

    assert surf_mesh.connect is not None
    assert surf_mesh.connect["connect1"].shape == (3, 4)
    assert np.min(surf_mesh.connect["connect1"]) == 1
