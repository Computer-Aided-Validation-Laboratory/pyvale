import numpy as np
import pytest

from pyvale import dataio, vfm
from pyvale.vfm.roi import convert_mask_to_physical_roi


@pytest.mark.parametrize("grid_shape", [None, (2, 3)])
def test_build_vfm_input_data_places_strain_and_force(
    grid_shape: tuple[int, int] | None,
) -> None:
    x_grid, y_grid = np.meshgrid([0.0, 2.0, 4.0], [0.0, 3.0])
    coordinates = np.column_stack((x_grid.ravel(), y_grid.ravel(), np.zeros(6)))
    times = np.array([0.0, 2.0])
    # Distinct point, component and time values expose axis transposition errors.
    fields = np.arange(48, dtype=float).reshape(6, 4, 2)
    fields[1] = np.nan
    order = [4, 0, 3, 5, 2, 1] if grid_shape is None else list(range(6))
    dic_data = dataio.ExpData(
        fields[order], {}, {}, coordinates[order], times,
        component_labels=("strain_xy", "disp_x", "strain_xx", "strain_yy"),
        grid_shape=grid_shape,
    )
    daq_data = dataio.ExpData(
        np.array([[4.0, 8.0]]), {"load": 0}, {0: "load"}, times=times
    )
    mask = np.isfinite(fields[:, 0, 0]).reshape(x_grid.shape)
    roi = vfm.VfmRegionOfInterest.from_definition(
        convert_mask_to_physical_roi(mask, x_grid, y_grid, simplification_pixels=0.0)
    )
    free = vfm.Edge(vfm.EEdgeCondition.Free, vfm.EEdgeCondition.Free)
    edges = vfm.EdgeConditions(free, free, free, free)

    vfm_input = vfm.build_vfm_input_data(
        dic_data, daq_data, thickness=2.0, edge_conditions=edges,
        region_of_interest=roi, force_y_label="load",
        shear_convention=dataio.EStrainShear.ENGINEERING,
    )

    expected = fields[:, [2, 3, 0], :].transpose(2, 1, 0).reshape(2, 3, 2, 3)
    expected[:, 2] *= 0.5
    assert isinstance(vfm_input, vfm.VfmInputData)
    np.testing.assert_array_equal(vfm_input.strain, expected)
    np.testing.assert_array_equal(
        vfm_input.specimen_geometry.pixel_area, np.full((2, 3), 6)
    )
    np.testing.assert_array_equal(vfm_input.boundary_conditions.force, [[0, 4], [0, 8]])
    np.testing.assert_array_equal(vfm_input.delta_timesteps, [0, 2])

    daq_data.times = np.array([0.0, 3.0])
    with pytest.raises(ValueError, match="DAQ timestamps must match"):
        vfm.build_vfm_input_data(
            dic_data, daq_data, thickness=2.0, edge_conditions=edges,
            region_of_interest=roi, force_y_label="load",
        )
