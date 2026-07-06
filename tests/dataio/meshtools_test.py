# ==============================================================================
# pyvale: the python validation engine
# License: MIT
# Copyright (C) 2025 The Computer Aided Validation Team
# ==============================================================================

import numpy as np
import pytest

import pyvale.dataio as io
import pyvale.mooseherder as mh
import pyvale.dataset as dataset


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


def _calc_face_normal(face_coords: np.ndarray) -> np.ndarray:
    if face_coords.shape[0] in (3, 6, 7):
        corners = face_coords[:3, :]
    else:
        corners = face_coords[:4, :]

    face_normal = np.cross(
        corners[1] - corners[0],
        corners[2] - corners[0],
    )
    normal_mag = np.linalg.norm(face_normal)

    if normal_mag <= 1.0e-12 and corners.shape[0] == 4:
        face_normal = np.cross(
            corners[2] - corners[0],
            corners[3] - corners[0],
        )
        normal_mag = np.linalg.norm(face_normal)

    assert normal_mag > 1.0e-12
    return face_normal / normal_mag


def _assert_extracted_surface_points_outward(surf: io.SimData) -> None:
    assert surf.connect is not None
    assert surf.coords is not None

    mesh_centroid = np.mean(surf.coords, axis=0)

    for connect in surf.connect.values():
        for face_connect in connect:
            face_coords = surf.coords[face_connect]
            face_normal = _calc_face_normal(face_coords)

            if face_coords.shape[0] in (3, 6, 7):
                face_centroid = np.mean(face_coords[:3, :], axis=0)
            else:
                face_centroid = np.mean(face_coords[:4, :], axis=0)

            outward_dir = face_centroid - mesh_centroid
            assert np.dot(face_normal, outward_dir) > 0.0


def _assert_higher_order_surface_edge_order(surf: io.SimData) -> None:
    assert surf.connect is not None
    assert surf.coords is not None

    for connect in surf.connect.values():
        nodes_per_face = connect.shape[1]

        if nodes_per_face == 6:
            edge_corner_pairs = ((0, 1), (1, 2), (2, 0))
            mid_inds = (3, 4, 5)
        elif nodes_per_face == 7:
            edge_corner_pairs = ((0, 1), (1, 2), (2, 0))
            mid_inds = (3, 4, 5)
        elif nodes_per_face == 8:
            edge_corner_pairs = ((0, 1), (1, 2), (2, 3), (3, 0))
            mid_inds = (4, 5, 6, 7)
        elif nodes_per_face == 9:
            edge_corner_pairs = ((0, 1), (1, 2), (2, 3), (3, 0))
            mid_inds = (4, 5, 6, 7)
        else:
            continue

        for face_connect in connect:
            face_coords = surf.coords[face_connect]
            edge_midpoints = np.array(
                [
                    0.5 * (
                        face_coords[start_ind, :] +
                        face_coords[end_ind, :]
                    )
                    for (start_ind, end_ind) in edge_corner_pairs
                ],
                dtype=np.float64,
            )
            midside_coords = face_coords[np.array(mid_inds, dtype=np.int64), :]
            edge_dists = np.linalg.norm(
                midside_coords[:, None, :] - edge_midpoints[None, :, :],
                axis=2,
            )
            expected_mid_order = np.argmin(edge_dists, axis=0)
            assert np.array_equal(
                expected_mid_order,
                np.arange(len(mid_inds), dtype=np.int64),
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


def test_extract_surf_between_center_slice_hex8() -> None:
    path = dataset.element_case_output_path(dataset.EElemTest.HEX8)
    mesh = mh.ExodusLoader(path).load_all_sim_data()

    # Slice at z = 0.005 (center line)
    surf = io.extract_surf_between(
        mesh,
        point=(0.0, 0.0, 0.005),
        normal=(0.0, 0.0, 1.0),
        distance=None,
        tolerance=1.0e-6,
    )

    assert surf.connect is not None
    assert "connect1" in surf.connect
    # 4 quad elements, each has 4 nodes
    assert surf.connect["connect1"].shape == (4, 4)
    # 9 nodes in a 3x3 grid
    assert surf.coords.shape == (9, 3)
    # Check Z coordinates are all exactly 0.005
    assert np.all(np.abs(surf.coords[:, 2] - 0.005) < 1.0e-6)


def test_extract_surf_between_boundary_slice_hex8() -> None:
    path = dataset.element_case_output_path(dataset.EElemTest.HEX8)
    mesh = mh.ExodusLoader(path).load_all_sim_data()

    # Slice at z = 0.01 (top boundary)
    surf = io.extract_surf_between(
        mesh,
        point=(0.0, 0.0, 0.01),
        normal=(0.0, 0.0, 1.0),
        distance=None,
        tolerance=1.0e-6,
    )

    assert surf.connect is not None
    assert "connect1" in surf.connect
    assert surf.connect["connect1"].shape == (4, 4)
    assert surf.coords.shape == (9, 3)
    assert np.all(np.abs(surf.coords[:, 2] - 0.01) < 1.0e-6)


def test_extract_surf_between_slab_hex8() -> None:
    path = dataset.element_case_output_path(dataset.EElemTest.HEX8)
    mesh = mh.ExodusLoader(path).load_all_sim_data()

    # Slice from z = 0.005 to z = 0.01 (top half slab)
    surf = io.extract_surf_between(
        mesh,
        point=(0.0, 0.0, 0.005),
        normal=(0.0, 0.0, 1.0),
        distance=0.005,
        tolerance=1.0e-6,
    )

    assert surf.connect is not None
    assert "connect1" in surf.connect
    # 20 unique faces inside the top half
    assert surf.connect["connect1"].shape == (20, 4)
    # 18 nodes total (9 at z=0.005 and 9 at z=0.01)
    assert surf.coords.shape == (18, 3)


def test_extract_surf_between_raises_when_empty() -> None:
    path = dataset.element_case_output_path(dataset.EElemTest.HEX8)
    mesh = mh.ExodusLoader(path).load_all_sim_data()

    # Try to slice at z = 0.05 (outside the 0.0 to 0.01 cube)
    with pytest.raises(ValueError, match="No elements/faces found"):
        io.extract_surf_between(
            mesh,
            point=(0.0, 0.0, 0.05),
            normal=(0.0, 0.0, 1.0),
            distance=None,
            tolerance=1.0e-6,
        )


def test_extract_surf_between_boundary_slice_tet4() -> None:
    path = dataset.element_case_output_path(dataset.EElemTest.TET4)
    mesh = mh.ExodusLoader(path).load_all_sim_data()

    # Slice at z = 0.0 (bottom boundary)
    surf = io.extract_surf_between(
        mesh,
        point=(0.0, 0.0, 0.0),
        normal=(0.0, 0.0, 1.0),
        distance=None,
        tolerance=1.0e-6,
    )

    assert surf.connect is not None
    assert "connect1" in surf.connect
    # 4 tri elements, each has 3 nodes
    assert surf.connect["connect1"].shape == (4, 3)
    assert surf.coords.shape == (5, 3)
    assert np.all(np.abs(surf.coords[:, 2] - 0.0) < 1.0e-6)


def test_extract_surf_between_boundary_slice_hex27() -> None:
    path = dataset.element_case_output_path(dataset.EElemTest.HEX27)
    mesh = mh.ExodusLoader(path).load_all_sim_data()

    # Slice at z = 0.01 (top boundary)
    surf = io.extract_surf_between(
        mesh,
        point=(0.0, 0.0, 0.01),
        normal=(0.0, 0.0, 1.0),
        distance=None,
        tolerance=1.0e-6,
    )

    assert surf.connect is not None
    assert "connect1" in surf.connect
    # 4 quad elements, each has 9 nodes (quadratic HEX27 faces are QUAD9)
    assert surf.connect["connect1"].shape == (4, 9)
    # 25 nodes in a 5x5 grid
    assert surf.coords.shape == (25, 3)
    assert np.all(np.abs(surf.coords[:, 2] - 0.01) < 1.0e-6)


def test_extract_surf_mesh_tet4() -> None:
    path = dataset.element_case_output_path(dataset.EElemTest.TET4)
    mesh = mh.ExodusLoader(path).load_all_sim_data()
    surf = io.extract_surf_mesh(mesh)
    assert surf.connect is not None
    assert surf.connect["connect1"].shape == (24, 3)
    assert surf.coords.shape == (14, 3)
    _assert_extracted_surface_points_outward(surf)


def test_extract_surf_mesh_tet10() -> None:
    path = dataset.element_case_output_path(dataset.EElemTest.TET10)
    mesh = mh.ExodusLoader(path).load_all_sim_data()
    surf = io.extract_surf_mesh(mesh)
    assert surf.connect is not None
    assert surf.connect["connect1"].shape == (24, 6)
    assert surf.coords.shape == (50, 3)
    _assert_extracted_surface_points_outward(surf)
    _assert_higher_order_surface_edge_order(surf)


def test_extract_surf_mesh_hex8() -> None:
    path = dataset.element_case_output_path(dataset.EElemTest.HEX8)
    mesh = mh.ExodusLoader(path).load_all_sim_data()
    surf = io.extract_surf_mesh(mesh)
    assert surf.connect is not None
    assert surf.connect["connect1"].shape == (24, 4)
    assert surf.coords.shape == (26, 3)
    _assert_extracted_surface_points_outward(surf)


def test_extract_surf_mesh_hex20() -> None:
    path = dataset.element_case_output_path(dataset.EElemTest.HEX20)
    mesh = mh.ExodusLoader(path).load_all_sim_data()
    surf = io.extract_surf_mesh(mesh)
    assert surf.connect is not None
    assert surf.connect["connect1"].shape == (24, 8)
    assert surf.coords.shape == (74, 3)
    _assert_extracted_surface_points_outward(surf)
    _assert_higher_order_surface_edge_order(surf)


def test_extract_surf_mesh_hex27() -> None:
    path = dataset.element_case_output_path(dataset.EElemTest.HEX27)
    mesh = mh.ExodusLoader(path).load_all_sim_data()
    surf = io.extract_surf_mesh(mesh)
    assert surf.connect is not None
    assert surf.connect["connect1"].shape == (24, 9)
    assert surf.coords.shape == (98, 3)
    _assert_extracted_surface_points_outward(surf)
    _assert_higher_order_surface_edge_order(surf)

