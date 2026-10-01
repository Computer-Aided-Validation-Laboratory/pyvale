from __future__ import annotations

"""Tests for the equilibrium-gap indicator on explicit synthetic fixtures."""

import numpy as np
import pytest

from pyvale.vfm.experimentdata import (
    BoundaryConditions,
    Edge,
    EdgeConditions,
    EEdgeCondition,
    ExperimentData,
    SpecimenGeometry,
)
from pyvale.vfm.metricequilibriumgap import (
    EquilibriumGapMetric,
    EquilibriumGapVirtualFieldType,
)
from pyvale.vfm.roi import RoiDefinition, RoiShape, VfmRegionOfInterest


def _rectangle_experiment_data(
    *,
    rows: int = 17,
    cols: int = 19,
    forces: tuple[float, ...] = (10.0, -20.0, 40.0),
    thickness: float = 2.0,
    pixel_area: float = 0.5,
) -> ExperimentData:
    x_coordinates, y_coordinates = np.meshgrid(
        np.arange(cols, dtype=np.float64),
        np.arange(rows, dtype=np.float64),
    )
    region_of_interest = VfmRegionOfInterest.from_definition(
        RoiDefinition(
            shapes=(
                RoiShape(
                    shape_type="rectangle",
                    index=0,
                    is_cutting=False,
                    rectangle=(0.0, 0.0, float(cols - 1), float(rows - 1)),
                ),
            ),
        )
    )
    specimen_geometry = SpecimenGeometry(
        x_coordinates,
        y_coordinates,
        np.full((rows, cols), pixel_area, dtype=np.float64),
        thickness,
        region_of_interest,
    )
    boundary_conditions = BoundaryConditions(
        EdgeConditions(
            min_x_edge=Edge(EEdgeCondition.Fixed, EEdgeCondition.Free),
            max_x_edge=Edge(EEdgeCondition.Traction, EEdgeCondition.Free),
            min_y_edge=Edge(EEdgeCondition.Free, EEdgeCondition.Free),
            max_y_edge=Edge(EEdgeCondition.Free, EEdgeCondition.Free),
        ),
        np.column_stack(
            (
                np.asarray(forces, dtype=np.float64),
                np.zeros(len(forces), dtype=np.float64),
            )
        ),
    )
    strain = np.zeros((len(forces), 3, rows, cols), dtype=np.float64)
    timesteps = np.arange(len(forces), dtype=np.float64)
    return ExperimentData(strain, specimen_geometry, boundary_conditions, timesteps)


def _constant_stress(
    experiment_data: ExperimentData,
    *,
    sigma_xx: float = 150.0,
    sigma_yy: float = 40.0,
    sigma_xy: float = -25.0,
) -> np.ndarray:
    rows, cols = experiment_data.specimen_geometry.x.shape
    steps = experiment_data.timesteps.size
    stress = np.zeros((steps, 3, rows, cols), dtype=np.float64)
    stress[:, 0, :, :] = sigma_xx
    stress[:, 1, :, :] = sigma_yy
    stress[:, 2, :, :] = sigma_xy
    return stress


def _stress_with_central_inclusion(
    experiment_data: ExperimentData,
    *,
    scale: float = 4.0,
) -> np.ndarray:
    stress = _constant_stress(experiment_data)
    rows, cols = experiment_data.specimen_geometry.x.shape
    stress[:, :, rows // 2, cols // 2] *= scale
    return stress

# Run test 3 times with different virtual field types to ensure
# that the equilibrium gap metric behaves correctly for each type.
@pytest.mark.parametrize(
    "virtual_field_type",
    (
        EquilibriumGapVirtualFieldType.SINGLE_POS_POS,
        EquilibriumGapVirtualFieldType.SINGLE_POS_NEG,
        EquilibriumGapVirtualFieldType.TWO_AVERAGED,
    ),
)
def test_constant_stress_field_has_zero_equilibrium_gap(
    virtual_field_type: EquilibriumGapVirtualFieldType,
) -> None:
    """A divergence-free constant stress field should do no internal virtual work."""

    experiment_data = _rectangle_experiment_data()
    metric = EquilibriumGapMetric(
        window_size=(5, 5),
        valid_window_fill_fraction=1.0,
        virtual_field_type=virtual_field_type,
    )
    metric.initialise(experiment_data)

    result = metric.evaluate_equilibrium_gap(_constant_stress(experiment_data))

    assert result.weighted_spatiotemporal_rms is not None
    assert result.weighted_spatiotemporal_rms < 1.0e-12
    assert np.nanmax(np.abs(result.normalised_gap)) < 1.0e-12


def test_local_stress_inconsistency_creates_local_equilibrium_gap_hotspot() -> None:
    """A local stress perturbation should be detected near the perturbation."""

    experiment_data = _rectangle_experiment_data()
    metric = EquilibriumGapMetric(
        window_size=(5, 5),
        valid_window_fill_fraction=1.0,
    )
    metric.initialise(experiment_data)

    baseline = metric.evaluate_equilibrium_gap(_constant_stress(experiment_data))
    perturbed = metric.evaluate_equilibrium_gap(
        _stress_with_central_inclusion(experiment_data)
    )

    assert baseline.weighted_spatiotemporal_rms is not None
    assert perturbed.weighted_spatiotemporal_rms is not None
    assert perturbed.weighted_spatiotemporal_rms > 1.0e-3
    assert perturbed.weighted_spatiotemporal_rms > (
        1.0e9 * baseline.weighted_spatiotemporal_rms
    )

    last_frame = np.abs(perturbed.normalised_gap[-1])
    max_index = np.unravel_index(np.nanargmax(last_frame), last_frame.shape)
    rows, cols = experiment_data.specimen_geometry.x.shape
    assert abs(max_index[0] - rows // 2) <= 1
    assert abs(max_index[1] - cols // 2) <= 1


def test_normalised_gap_is_raw_gap_divided_by_force_and_window_count() -> None:
    """The reported load-step normalisation should follow the EGI definition."""

    experiment_data = _rectangle_experiment_data()
    metric = EquilibriumGapMetric(
        window_size=(5, 5),
        valid_window_fill_fraction=1.0,
    )
    metric.initialise(experiment_data)

    result = metric.evaluate_equilibrium_gap(
        _stress_with_central_inclusion(experiment_data)
    )
    metadata = result.metric_result.additional_fields
    force = np.abs(experiment_data.boundary_conditions.force[:, 0])
    denominator = (
        force[:, np.newaxis, np.newaxis]
        * metadata["nominal_window_point_count"]
    )
    expected = result.raw_gap / denominator

    valid_values = np.isfinite(result.normalised_gap)
    np.testing.assert_allclose(
        result.normalised_gap[valid_values],
        expected[valid_values],
        rtol=1.0e-12,
        atol=1.0e-12,
    )
    assert np.all(np.isnan(result.normalised_gap[:, ~metadata["valid_centre_mask"]]))


def test_common_stress_and_force_scaling_leaves_normalised_gap_unchanged() -> None:
    """EGI should not distinguish stress fields that differ only by a load scale."""

    scale = 3.5
    experiment_data = _rectangle_experiment_data()
    scaled_experiment_data = _rectangle_experiment_data(
        forces=tuple(scale * experiment_data.boundary_conditions.force[:, 0])
    )
    stress = _stress_with_central_inclusion(experiment_data)
    scaled_stress = scale * stress

    metric = EquilibriumGapMetric(
        window_size=(5, 5),
        valid_window_fill_fraction=1.0,
    )
    scaled_metric = EquilibriumGapMetric(
        window_size=(5, 5),
        valid_window_fill_fraction=1.0,
    )
    metric.initialise(experiment_data)
    scaled_metric.initialise(scaled_experiment_data)

    result = metric.evaluate_equilibrium_gap(stress)
    scaled_result = scaled_metric.evaluate_equilibrium_gap(scaled_stress)

    valid_values = np.isfinite(result.normalised_gap)
    np.testing.assert_allclose(
        scaled_result.raw_gap[valid_values],
        scale * result.raw_gap[valid_values],
        rtol=1.0e-12,
        atol=1.0e-12,
    )
    np.testing.assert_allclose(
        scaled_result.normalised_gap[valid_values],
        result.normalised_gap[valid_values],
        rtol=1.0e-12,
        atol=1.0e-12,
    )


def test_windows_can_cross_free_edges_but_not_non_free_edges() -> None:
    """Free edges may use clipped windows; fixed and traction edges lose a half-window border."""

    experiment_data = _rectangle_experiment_data()
    metric = EquilibriumGapMetric(
        window_size=(5, 5),
        valid_window_fill_fraction=0.0,
    )
    metric.initialise(experiment_data)
    result = metric.evaluate_equilibrium_gap(_constant_stress(experiment_data))
    valid_centre_mask = result.metric_result.additional_fields["valid_centre_mask"]

    valid_rows, valid_cols = np.where(valid_centre_mask)

    assert valid_rows.min() == 0
    assert valid_rows.max() == experiment_data.specimen_geometry.x.shape[0] - 1
    assert valid_cols.min() == 2
    assert valid_cols.max() == experiment_data.specimen_geometry.x.shape[1] - 3
