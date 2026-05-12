from __future__ import annotations

from pathlib import Path

import numpy as np

from pyvale.vfm.project_definition import (
    BoundaryConditions,
    EdgeBoundaryCondition,
    EEdgeBoundaryCondition,
)
from pyvale.vfm.virtual_fields_mesh import (
    generate_vf_from_mesh,
    generate_virtual_fields_mesh,
)


def _build_boundary_conditions() -> BoundaryConditions:
    """Return the boundary conditions used by the regression fixture."""

    return BoundaryConditions(
        min_x_edge=EdgeBoundaryCondition(
            x=EEdgeBoundaryCondition.FIXED,
            y=EEdgeBoundaryCondition.FIXED,
        ),
        max_y_edge=EdgeBoundaryCondition(
            x=EEdgeBoundaryCondition.FREE,
            y=EEdgeBoundaryCondition.FREE,
        ),
        max_x_edge=EdgeBoundaryCondition(
            x=EEdgeBoundaryCondition.TRACTION,
            y=EEdgeBoundaryCondition.FIXED,
        ),
        min_y_edge=EdgeBoundaryCondition(
            x=EEdgeBoundaryCondition.FREE,
            y=EEdgeBoundaryCondition.FREE,
        ),
    )


def main() -> None:
    """Generate the lightweight regression fixture used by test_virtual_fields_mesh.py."""

    # Keep the datapoint grid modest enough for git, but large enough to exercise
    # row/column ordering. The virtual-fields mesh is intentionally asymmetric.
    x_coordinates = np.linspace(0.5, 14.5, 15, dtype=np.float64)
    y_coordinates = np.linspace(0.5, 9.5, 10, dtype=np.float64)
    x = np.tile(x_coordinates[np.newaxis, :], (y_coordinates.size, 1))
    y = np.tile(y_coordinates[:, np.newaxis], (1, x_coordinates.size))
    specimen_mask = np.ones((10, 15), dtype=bool)
    mesh_size = np.array([3, 5], dtype=np.uint32)

    mesh = generate_virtual_fields_mesh(
        x=x,
        y=y,
        specimen_mask=specimen_mask,
        boundary_conditions=_build_boundary_conditions(),
        mesh_size=mesh_size,
        generate_plots=False,
    )

    reference_map = np.empty((2, 3, 10, 15), dtype=np.float64)
    reference_map[0, 0] = 10.0 + 0.5 * x + 0.25 * y
    reference_map[0, 1] = -3.0 + 0.2 * x - 0.1 * y
    reference_map[0, 2] = 0.1 * np.sin(x) + 0.2 * np.cos(y)
    reference_map[1, 0] = 11.0 + 0.4 * x + 0.3 * y
    reference_map[1, 1] = -2.0 + 0.15 * x - 0.05 * y
    reference_map[1, 2] = 0.12 * np.sin(0.5 * x) - 0.08 * np.cos(1.5 * y)

    virtual_fields = generate_vf_from_mesh(
        reference_map,
        mesh,
        plot_fields=False,
    )

    fixture_path = Path(__file__).with_name("fixtures") / "virtual_fields_mesh_regression_case.npz"
    fixture_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        fixture_path,
        x=x,
        y=y,
        specimen_mask=specimen_mask,
        mesh_size=mesh_size,
        reference_map=reference_map,
        virtual_node_coordinates_x=mesh.virtual_node_coordinates_x,
        virtual_node_coordinates_y=mesh.virtual_node_coordinates_y,
        specimen_point_indices=mesh.specimen_point_indices,
        virtual_node_ids=mesh.virtual_node_ids,
        virtual_element_node_ids=mesh.virtual_element_node_ids,
        data_point_virtual_element_ids=mesh.data_point_virtual_element_ids,
        global_shape_function_matrix=mesh.global_shape_function_matrix,
        global_strain_displacement_matrix=mesh.global_strain_displacement_matrix,
        global_strain_displacement_matrix_pseudoinverse=mesh.global_strain_displacement_matrix_pseudoinverse,
        active_degrees_of_freedom=mesh.active_degrees_of_freedom,
        virtual_strain=virtual_fields.virtual_strain,
        virtual_displacement_edge=virtual_fields.virtual_displacement_edge,
        virtual_displacement=virtual_fields.virtual_displacement,
    )
    print(f"Saved fixture to {fixture_path}")


if __name__ == "__main__":
    main()
