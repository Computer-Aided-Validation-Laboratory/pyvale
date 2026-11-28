# ==============================================================================
# pyvale: the python validation engine
# License: MIT
# Copyright (C) 2025 The Computer Aided Validation Team
# ==============================================================================

"""Analytic field samplers through the point sensor factories."""

import numpy as np
import numpy.typing as npt
import pytest
from scipy.spatial.transform import Rotation

import pyvale.sensorsim as sens
from pyvale.verif.analyticsimdatafactory import scalar_linear_2d


def evaluate_temperature(
    points: npt.NDArray[np.float64], times: npt.NDArray[np.float64]
) -> npt.NDArray[np.float64]:
    return (20.0 + points[:, 0, None] + times[None, :] ** 2)[:, None, :]


def test_factory_samples_analytic_scalar_with_inclusive_bounds() -> None:
    def evaluate_in_bounds(
        points: npt.NDArray[np.float64], times: npt.NDArray[np.float64]
    ) -> npt.NDArray[np.float64]:
        assert np.all((0.0 <= points[:, 0]) & (points[:, 0] <= 2.0))
        assert np.all((0.0 <= times) & (times <= 1.0))
        return evaluate_temperature(points, times)

    sampler = sens.FieldSampleAnalytic(
        evaluate=evaluate_in_bounds,
        default_times=np.array([0.0, 1.0]),
        num_components=1,
        bounds=sens.FieldSampleBounds(x=(0.0, 2.0), time=(0.0, 1.0)),
    )
    sensors = sens.SensorFactory.scalar_point(
        field_source=sampler,
        sensor_data=sens.SensorData(
            positions=np.array([[2.0, 0.0, 0.0], [3.0, 0.0, 0.0]]),
            sample_times=np.array([0.0, 0.5, 1.0, 1.5]),
        ),
        comp_key="temperature",
        spatial_dims=sens.EDim.TWOD,
    )
    np.testing.assert_allclose(
        sensors.get_truth(), [[[22.0, 22.25, 23.0, np.nan]], [[np.nan] * 4]]
    )
    np.testing.assert_allclose(
        sampler.sample(np.array([[0.0, 0.0, 0.0]])), [[[20.0, 21.0]]]
    )
    assert np.isnan(sampler.sample(np.array([[3.0, 0.0, 0.0]]))).all()


def evaluate_vector(
    points: npt.NDArray[np.float64], times: npt.NDArray[np.float64]
) -> npt.NDArray[np.float64]:
    return np.broadcast_to(
        np.array([2.0, 3.0])[None, :, None], (len(points), 2, len(times))
    )


def evaluate_tensor(
    points: npt.NDArray[np.float64], times: npt.NDArray[np.float64]
) -> npt.NDArray[np.float64]:
    return np.broadcast_to(
        np.array([2.0, 3.0, 1.0])[None, :, None], (len(points), 3, len(times))
    )


def test_factory_rotates_analytic_vector_and_tensor_components() -> None:
    data = sens.SensorData(
        positions=np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]]),
        angles=(Rotation.identity(), Rotation.from_euler("z", 90, degrees=True)),
    )
    times = np.array([0.0])
    vector = sens.SensorFactory.vector_point(
        sens.FieldSampleAnalytic(
            evaluate=evaluate_vector, default_times=times, num_components=2
        ),
        data,
        comp_keys=("x", "y"),
        spatial_dims=sens.EDim.TWOD,
    )
    tensor = sens.SensorFactory.tensor_point(
        sens.FieldSampleAnalytic(
            evaluate=evaluate_tensor, default_times=times, num_components=3
        ),
        data,
        norm_comp_keys=("xx", "yy"),
        dev_comp_keys=("xy",),
        spatial_dims=sens.EDim.TWOD,
    )
    np.testing.assert_allclose(vector.get_truth()[..., 0], [[2.0, 3.0], [3.0, -2.0]])
    np.testing.assert_allclose(
        tensor.get_truth()[..., 0], [[2.0, 3.0, 1.0], [3.0, 2.0, -1.0]]
    )


def test_replacing_sampler_clears_caches_and_refreshes_time_shape() -> None:
    sim_data, _ = scalar_linear_2d()
    positions = np.array([[1.0, 1.0, 0.0]])
    sensors = sens.SensorFactory.scalar_point(
        sim_data,
        sens.SensorData(positions=positions),
        comp_key="temperature",
        spatial_dims=sens.EDim.TWOD,
    )
    sensors.set_error_chain([sens.ErrSysOffset(offset=2.0)])
    before = sensors.sim_measurements().copy()
    times = np.array([0.0, 0.25, 0.5])

    sensors.set_field_sampler(
        sens.FieldSampleAnalytic(
            evaluate=evaluate_temperature, default_times=times, num_components=1
        )
    )
    assert sensors.get_measurement_shape() == (1, 1, 3)
    np.testing.assert_array_equal(sensors.get_errors_total(), np.zeros((1, 1, 3)))
    np.testing.assert_allclose(
        sensors.get_measurements(), (23.0 + times**2)[None, None]
    )
    with pytest.raises(ValueError, match="component count"):
        sensors.set_field_sampler(
            sens.FieldSampleAnalytic(
                evaluate=evaluate_vector, default_times=times, num_components=2
            )
        )
    np.testing.assert_allclose(sensors.get_truth(), (21.0 + times**2)[None, None])

    sensors.set_field_sampler(
        sens.FieldInterpMesh(sim_data, ("temperature",), sens.EDim.TWOD)
    )
    np.testing.assert_allclose(sensors.get_measurements(), before)
