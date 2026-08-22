import numpy as np

from pyvale.vfm.metricsliceforce import (
    build_force_reconstruction_error_result,
    compute_force_temporal_weights,
)


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
