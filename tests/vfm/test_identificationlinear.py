"""Analytical checks for direct homogeneous linear-elastic VFM identification."""

from __future__ import annotations

import numpy as np

from pyvale.vfm import (
    BoundaryConditions,
    Edge,
    EdgeConditions,
    EEdgeCondition,
    ExperimentData,
    SpecimenGeometry,
    VfmRegionOfInterest,
    convert_mask_to_physical_roi,
    identify_isotropic_linear_elasticity_plane_stress,
)


def test_direct_plane_stress_solve_recovers_analytical_uniaxial_material() -> None:
    elastic_modulus = 210_000.0
    poissons_ratio = 0.3
    thickness = 0.8
    spacing = 0.5
    rows, columns = 20, 10
    x_axis = spacing * np.arange(columns)
    y_axis = spacing * np.arange(rows)
    x, y = np.meshgrid(x_axis, y_axis)
    mask = np.ones((rows, columns), dtype=bool)
    roi = VfmRegionOfInterest.from_definition(
        convert_mask_to_physical_roi(mask, x, y, simplification_pixels=0.0)
    )

    axial_strain = np.asarray((0.0004, 0.0008, 0.0012, 0.0016))
    strain = np.zeros((axial_strain.size, 3, rows, columns))
    strain[:, 0] = -poissons_ratio * axial_strain[:, np.newaxis, np.newaxis]
    strain[:, 1] = axial_strain[:, np.newaxis, np.newaxis]
    force = np.zeros((axial_strain.size, 2))
    represented_width = columns * spacing
    force[:, 1] = elastic_modulus * axial_strain * represented_width * thickness
    free = Edge(EEdgeCondition.Free, EEdgeCondition.Free)
    edge_conditions = EdgeConditions(
        min_x_edge=free,
        max_x_edge=free,
        min_y_edge=Edge(EEdgeCondition.Fixed, EEdgeCondition.Fixed),
        max_y_edge=Edge(EEdgeCondition.Free, EEdgeCondition.Traction),
    )
    data = ExperimentData(
        strain=strain,
        specimen_geometry=SpecimenGeometry(
            x=x,
            y=y,
            pixel_area=np.full_like(x, spacing**2),
            thickness=thickness,
            region_of_interest=roi,
        ),
        boundary_conditions=BoundaryConditions(edge_conditions, force),
        timesteps=np.arange(axial_strain.size, dtype=np.float64),
    )

    result = identify_isotropic_linear_elasticity_plane_stress(
        data,
        virtual_mesh_size=(4, 4),
        spatial_stride=(2, 2),
    )

    assert result.rank == 2
    assert result.spatial_stride == (2, 2)
    assert result.analysis_grid_shape == (10, 5)
    assert result.scaled_residual_norm < 1.0e-12
    assert np.isclose(result.elastic_modulus, elastic_modulus, rtol=1.0e-10)
    assert np.isclose(result.poissons_ratio, poissons_ratio, atol=1.0e-12)
