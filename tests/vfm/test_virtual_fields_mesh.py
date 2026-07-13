# from __future__ import annotations

# from pathlib import Path

# import numpy as np
# import numpy.testing as nptest

# from pyvale.vfm.project_definition import (
#     BoundaryConditions,
#     EdgeBoundaryCondition,
#     EEdgeBoundaryCondition,
# )
# from pyvale.vfm.virtual_fields_mesh import (
#     generate_vf_from_mesh,
#     generate_virtual_fields_mesh,
# )


# def _fixture_path() -> Path:
#     """Return the lightweight regression fixture committed with the tests."""

#     return Path(__file__).with_name("fixtures") / "virtual_fields_mesh_regression_case.npz"


# def _load_regression_fixture() -> dict[str, np.ndarray]:
#     """Load the saved mesh and field snapshot used for regression tests."""

#     with np.load(_fixture_path()) as fixture_data:
#         return {name: fixture_data[name] for name in fixture_data.files}


# def _build_boundary_conditions() -> BoundaryConditions:
#     """Return one small but non-trivial boundary-condition set for the tests."""

#     return BoundaryConditions(
#         min_x_edge=EdgeBoundaryCondition(
#             x=EEdgeBoundaryCondition.FIXED,
#             y=EEdgeBoundaryCondition.FIXED,
#         ),
#         max_y_edge=EdgeBoundaryCondition(
#             x=EEdgeBoundaryCondition.FREE,
#             y=EEdgeBoundaryCondition.FREE,
#         ),
#         max_x_edge=EdgeBoundaryCondition(
#             x=EEdgeBoundaryCondition.TRACTION,
#             y=EEdgeBoundaryCondition.FIXED,
#         ),
#         min_y_edge=EdgeBoundaryCondition(
#             x=EEdgeBoundaryCondition.FREE,
#             y=EEdgeBoundaryCondition.FREE,
#         ),
#     )


# def test_generate_virtual_fields_mesh_rejects_y_decreasing_down_rows() -> None:
#     """The mesh generator should reject coordinate data with inverted y ordering."""

#     x = np.array([[0.5, 1.5], [0.5, 1.5]], dtype=np.float64)
#     y = np.array([[1.5, 1.5], [0.5, 0.5]], dtype=np.float64)
#     specimen_mask = np.ones((2, 2), dtype=bool)

#     with nptest.assert_raises_regex(
#         ValueError,
#         "y-coordinate should increase as array row increases",
#     ):
#         generate_virtual_fields_mesh(
#             x=x,
#             y=y,
#             specimen_mask=specimen_mask,
#             boundary_conditions=_build_boundary_conditions(),
#             mesh_size=np.array([1, 1], dtype=np.uint32),
#             generate_plots=False,
#         )


# def test_generate_virtual_fields_mesh_outputs_expected_shapes() -> None:
#     """The key mesh arrays should have the expected dimensions."""

#     fixture = _load_regression_fixture()
#     x = fixture["x"]
#     y = fixture["y"]
#     specimen_mask = fixture["specimen_mask"].astype(bool)
#     mesh_size = fixture["mesh_size"].astype(np.uint32)

#     mesh = generate_virtual_fields_mesh(
#         x=x,
#         y=y,
#         specimen_mask=specimen_mask,
#         boundary_conditions=_build_boundary_conditions(),
#         mesh_size=mesh_size,
#         generate_plots=False,
#     )

#     # The saved regression case uses an asymmetric 3 x 5 VF mesh on a 10 x 15
#     # datapoint grid so row/column swaps are easier to catch.
#     n_elements = int(mesh_size[0] * mesh_size[1])
#     n_nodes = int((mesh_size[0] + 1) * (mesh_size[1] + 1))
#     n_datapoints = x.size
#     n_specimen_points = int(np.sum(specimen_mask))

#     assert mesh.virtual_node_coordinates_x.shape == (mesh_size[0] + 1, mesh_size[1] + 1)
#     assert mesh.virtual_node_coordinates_y.shape == (mesh_size[0] + 1, mesh_size[1] + 1)
#     assert mesh.virtual_element_node_ids.shape == (n_elements, 4)
#     assert mesh.data_point_virtual_element_ids.shape == (n_datapoints,)
#     assert mesh.global_shape_function_matrix.shape == (n_specimen_points, n_nodes)
#     assert mesh.global_strain_displacement_matrix.shape == (3 * n_specimen_points, 2 * n_nodes)


# def test_generate_virtual_fields_mesh_matches_regression_fixture() -> None:
#     """The saved mesh snapshot should not change unless we intentionally update it."""

#     fixture = _load_regression_fixture()

#     mesh = generate_virtual_fields_mesh(
#         x=fixture["x"],
#         y=fixture["y"],
#         specimen_mask=fixture["specimen_mask"].astype(bool),
#         boundary_conditions=_build_boundary_conditions(),
#         mesh_size=fixture["mesh_size"].astype(np.uint32),
#         generate_plots=False,  #turn on for manual checking of plots when updating fixture
#     )

#     nptest.assert_array_equal(mesh.specimen_point_indices, fixture["specimen_point_indices"])
#     nptest.assert_array_equal(mesh.virtual_node_ids, fixture["virtual_node_ids"])
#     nptest.assert_array_equal(mesh.virtual_element_node_ids, fixture["virtual_element_node_ids"])
#     nptest.assert_array_equal(
#         mesh.data_point_virtual_element_ids,
#         fixture["data_point_virtual_element_ids"],
#     )
#     nptest.assert_array_equal(
#         mesh.active_degrees_of_freedom,
#         fixture["active_degrees_of_freedom"],
#     )

#     nptest.assert_allclose(
#         mesh.virtual_node_coordinates_x,
#         fixture["virtual_node_coordinates_x"],
#         rtol=1.0e-12,
#         atol=1.0e-12,
#     )
#     nptest.assert_allclose(
#         mesh.virtual_node_coordinates_y,
#         fixture["virtual_node_coordinates_y"],
#         rtol=1.0e-12,
#         atol=1.0e-12,
#     )
#     nptest.assert_allclose(
#         mesh.global_shape_function_matrix,
#         fixture["global_shape_function_matrix"],
#         rtol=1.0e-12,
#         atol=1.0e-12,
#     )
#     nptest.assert_allclose(
#         mesh.global_strain_displacement_matrix,
#         fixture["global_strain_displacement_matrix"],
#         rtol=1.0e-12,
#         atol=1.0e-12,
#     )
#     nptest.assert_allclose(
#         mesh.global_strain_displacement_matrix_pseudoinverse,
#         fixture["global_strain_displacement_matrix_pseudoinverse"],
#         rtol=1.0e-10,
#         atol=1.0e-10,
#     )


# def test_generate_vf_from_mesh_matches_regression_fixture() -> None:
#     """Generating fields from the saved reference map should match the saved snapshot."""

#     fixture = _load_regression_fixture()

#     mesh = generate_virtual_fields_mesh(
#         x=fixture["x"],
#         y=fixture["y"],
#         specimen_mask=fixture["specimen_mask"].astype(bool),
#         boundary_conditions=_build_boundary_conditions(),
#         mesh_size=fixture["mesh_size"].astype(np.uint32),
#         generate_plots=False, # turn on for manual checking of plots when updating fixture
#     )
#     virtual_fields = generate_vf_from_mesh(
#         fixture["reference_map"],
#         mesh,
#         plot_fields=False, # turn on for manual checking of plots when updating fixture
#     )

#     nptest.assert_allclose(
#         virtual_fields.virtual_strain,
#         fixture["virtual_strain"],
#         rtol=1.0e-10,
#         atol=1.0e-10,
#     )
#     nptest.assert_allclose(
#         virtual_fields.virtual_displacement,
#         fixture["virtual_displacement"],
#         rtol=1.0e-10,
#         atol=1.0e-10,
#     )
#     nptest.assert_allclose(
#         virtual_fields.virtual_displacement_edge,
#         fixture["virtual_displacement_edge"],
#         rtol=1.0e-10,
#         atol=1.0e-10,
#     )

# # Example usage:
# # uv run pytest tests/vfm/test_virtual_fields_mesh.py -q
