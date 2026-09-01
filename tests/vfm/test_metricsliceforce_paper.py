import numpy as np
import yaml

from pyvale.vfm.experimentdata import (
    BoundaryConditions,
    Edge,
    EdgeConditions,
    EEdgeCondition,
    ExperimentData,
    SpecimenGeometry,
)
from pyvale.vfm.metricsliceforce import (
    SliceWiseForceReconstructionMetric,
    build_force_reconstruction_error_result,
    compute_force_temporal_weights,
)
from pyvale.vfm.metricequilibriumgap import EquilibriumGapMetric
from pyvale.vfm.identificationresult import input_metadata_from_experiment_data
from pyvale.vfm.roi import RoiDefinition, RoiShape, VfmRegionOfInterest
from pyvale.vfm.slicewise_utils import SliceConfig


def test_force_temporal_weights_are_force_squared() -> None:
    weights = compute_force_temporal_weights(
        np.array([0.0, 10.0, -20.0])
    )

    np.testing.assert_allclose(weights, np.array([0.0, 0.2, 0.8]))


def test_force_reconstruction_uses_instantaneous_error_and_force_squared_weights() -> None:
    result = build_force_reconstruction_error_result(
        reconstructed_force=np.array([[11.0], [22.0]]),
        applied_longitudinal_force=np.array([10.0, 20.0]),
        temporal_weights=np.array([0.2, 0.8]),
        spatial_weights=np.array([1.0]),
    )

    # Both relative errors are 10%, so the weighted RMS is exactly 10%.
    assert result.weighted_spatiotemporal_rms == 0.1
    np.testing.assert_allclose(
        result.metric_result.additional_fields["normalised_residual"],
        np.array([[0.1], [0.1]]),
    )


def test_force_reconstruction_handles_negative_and_zero_force_steps() -> None:
    result = build_force_reconstruction_error_result(
        reconstructed_force=np.array([[-11.0], [99.0], [22.0]]),
        applied_longitudinal_force=np.array([-10.0, 0.0, 20.0]),
        temporal_weights=np.array([0.2, 0.0, 0.8]),
        spatial_weights=np.array([1.0]),
    )

    # The non-zero force steps both have 10% FRE; the zero-force step is masked.
    assert np.isclose(result.weighted_spatiotemporal_rms, 0.1)
    assert np.isnan(result.metric_result.additional_fields["normalised_residual"][1, 0])


def test_force_reconstruction_uses_slice_width_weights() -> None:
    result = build_force_reconstruction_error_result(
        reconstructed_force=np.array([[11.0, 14.0]]),
        applied_longitudinal_force=np.array([10.0]),
        temporal_weights=np.array([1.0]),
        spatial_weights=np.array([0.25, 0.75]),
    )

    assert np.isclose(result.weighted_spatiotemporal_rms, np.sqrt(0.25 * 0.1**2 + 0.75 * 0.4**2))


def _rectangle_roi(x_min: float, width: float) -> VfmRegionOfInterest:
    return VfmRegionOfInterest.from_definition(RoiDefinition(shapes=(
        RoiShape(
            shape_type="rectangle",
            index=0,
            is_cutting=False,
            rectangle=(x_min, -0.5, width, 2.0),
        ),
    )))


def _uniform_stress_experiment(*, corrected: bool) -> tuple[ExperimentData, np.ndarray]:
    x, y = np.meshgrid(np.array([-1.0, 0.0, 1.0]), np.array([0.0, 1.0]))
    measured_roi = _rectangle_roi(-1.5, 3.0)
    physical_roi = _rectangle_roi(-3.0, 6.0) if corrected else None
    geometry = SpecimenGeometry(
        x=x,
        y=y,
        pixel_area=np.ones_like(x),
        thickness=2.0,
        region_of_interest=measured_roi,
        force_reconstruction_region_of_interest=physical_roi,
    )
    free = Edge(EEdgeCondition.Free, EEdgeCondition.Free)
    experiment = ExperimentData(
        strain=np.zeros((1, 3, *x.shape)),
        specimen_geometry=geometry,
        boundary_conditions=BoundaryConditions(
            EdgeConditions(free, free, free, free),
            np.array([[0.0, 120.0]]),
        ),
        timesteps=np.array([1.0]),
    )
    stress = np.zeros_like(experiment.strain)
    stress[:, 1] = 10.0
    return experiment, stress


def test_optional_physical_roi_scales_fre_domain_without_changing_measurement_roi() -> None:
    uncorrected_experiment, stress = _uniform_stress_experiment(corrected=False)
    corrected_experiment, _ = _uniform_stress_experiment(corrected=True)
    metric = SliceWiseForceReconstructionMetric(
        slice_config=SliceConfig(axis="y", num_slices=1)
    )
    metric.initialise(uncorrected_experiment)

    uncorrected = metric.evaluate_force_recon_error(stress, uncorrected_experiment)
    corrected = metric.evaluate_force_recon_error(stress, corrected_experiment)
    uncorrected_fields = uncorrected.metric_result.additional_fields
    corrected_fields = corrected.metric_result.additional_fields

    np.testing.assert_allclose(uncorrected_fields["reconstructed_force"], [[60.0]])
    np.testing.assert_allclose(corrected_fields["reconstructed_force"], [[120.0]])
    np.testing.assert_allclose(corrected_fields["force_integration_scale_factors"], [2.0])
    np.testing.assert_allclose(corrected_fields["force_integration_measured_widths"], [3.0])
    np.testing.assert_allclose(corrected_fields["force_integration_target_widths"], [6.0])
    np.testing.assert_allclose(corrected_fields["force_integration_represented_fractions"], [0.5])
    assert corrected_fields["force_integration_domain_correction_enabled"] is True
    assert corrected.weighted_spatiotemporal_rms == 0.0
    np.testing.assert_array_equal(
        corrected_experiment.specimen_geometry.region_of_interest.sample_specimen_mask(
            corrected_experiment.specimen_geometry.x,
            corrected_experiment.specimen_geometry.y,
        ),
        np.ones((2, 3), dtype=bool),
    )


def test_physical_roi_scale_is_applied_to_fre_stress_adjoint() -> None:
    uncorrected_experiment, _ = _uniform_stress_experiment(corrected=False)
    corrected_experiment, _ = _uniform_stress_experiment(corrected=True)
    metric = SliceWiseForceReconstructionMetric(
        slice_config=SliceConfig(axis="y", num_slices=1)
    )
    metric.initialise(uncorrected_experiment)

    uncorrected = metric.normalised_residual_stress_adjoint(
        np.ones((1, 1)), uncorrected_experiment
    )
    corrected = metric.normalised_residual_stress_adjoint(
        np.ones((1, 1)), corrected_experiment
    )
    np.testing.assert_allclose(corrected[:, 1], 2.0 * uncorrected[:, 1])


def test_experiment_yaml_loads_and_records_optional_fre_roi(tmp_path) -> None:
    experiment, _ = _uniform_stress_experiment(corrected=True)
    np.save(tmp_path / "x.npy", experiment.specimen_geometry.x)
    np.save(tmp_path / "y.npy", experiment.specimen_geometry.y)
    np.save(tmp_path / "strain.npy", experiment.strain)
    np.save(tmp_path / "force.npy", experiment.boundary_conditions.force)
    np.save(tmp_path / "time.npy", experiment.timesteps)
    experiment.specimen_geometry.region_of_interest.save_yaml(
        tmp_path / "measured_roi.yaml"
    )
    assert experiment.specimen_geometry.force_reconstruction_region_of_interest is not None
    experiment.specimen_geometry.force_reconstruction_region_of_interest.save_yaml(
        tmp_path / "physical_roi.yaml"
    )
    manifest = {
        "x": "x.npy",
        "y": "y.npy",
        "strain": "strain.npy",
        "force": "force.npy",
        "time": "time.npy",
        "region_of_interest": "measured_roi.yaml",
        "force_reconstruction_region_of_interest": "physical_roi.yaml",
        "thickness": 2.0,
        "edge_conditions": {
            edge: {"x": "Free", "y": "Free"}
            for edge in ("min_x_edge", "max_x_edge", "min_y_edge", "max_y_edge")
        },
    }
    manifest_path = tmp_path / "experiment_data.yaml"
    manifest_path.write_text(yaml.safe_dump(manifest), encoding="utf-8")

    loaded = ExperimentData.load_from_file(manifest_path)
    metadata = input_metadata_from_experiment_data(
        loaded,
        source_path=manifest_path,
    )
    assert loaded.specimen_geometry.force_reconstruction_region_of_interest is not None
    assert metadata.force_reconstruction_domain_correction is True
    assert metadata.force_reconstruction_roi_source_path == str(
        (tmp_path / "physical_roi.yaml").resolve()
    )


def test_fre_physical_roi_does_not_change_egi() -> None:
    axis = np.arange(9, dtype=np.float64)
    x, y = np.meshgrid(axis, axis)
    measured_roi = VfmRegionOfInterest.from_definition(RoiDefinition(shapes=(
        RoiShape("rectangle", 0, False, rectangle=(-0.5, -0.5, 9.0, 9.0)),
    )))
    physical_roi = VfmRegionOfInterest.from_definition(RoiDefinition(shapes=(
        RoiShape("rectangle", 0, False, rectangle=(-2.0, -0.5, 12.0, 9.0)),
    )))
    free = Edge(EEdgeCondition.Free, EEdgeCondition.Free)
    traction = Edge(EEdgeCondition.Free, EEdgeCondition.Traction)
    strain = np.zeros((2, 3, 9, 9), dtype=np.float64)
    boundary = BoundaryConditions(
        EdgeConditions(free, free, free, traction),
        np.array([[0.0, 1.0], [0.0, 2.0]]),
    )
    base = ExperimentData(
        strain,
        SpecimenGeometry(x, y, np.ones_like(x), 1.0, measured_roi),
        boundary,
        np.array([0.0, 1.0]),
    )
    corrected = ExperimentData(
        strain.copy(),
        SpecimenGeometry(x, y, np.ones_like(x), 1.0, measured_roi, physical_roi),
        boundary,
        np.array([0.0, 1.0]),
    )
    rng = np.random.default_rng(42)
    stress = rng.normal(size=strain.shape)
    base_metric = EquilibriumGapMetric(window_size=(3, 3))
    corrected_metric = EquilibriumGapMetric(window_size=(3, 3))
    base_metric.initialise(base)
    corrected_metric.initialise(corrected)
    base_result = base_metric.evaluate_equilibrium_gap(stress).metric_result
    corrected_result = corrected_metric.evaluate_equilibrium_gap(stress).metric_result
    np.testing.assert_allclose(
        corrected_result.additional_fields["normalised_gap"],
        base_result.additional_fields["normalised_gap"],
        equal_nan=True,
    )
