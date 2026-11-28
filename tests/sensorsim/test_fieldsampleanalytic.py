# ==============================================================================
# pyvale: the python validation engine
# License: MIT
# Copyright (C) 2025 The Computer Aided Validation Team
# ==============================================================================

"""Analytic field samplers and descriptors for point sensor arrays."""

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


def test_analytic_scalar_with_inclusive_bounds() -> None:
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
    sensors_field = sens.FieldScalar(
        sampler,
        comp_key="temperature",
        spatial_dims=sens.EDim.TWOD,
    )
    sensors = sens.SensorsPoint(
        sensor_data=sens.SensorData(
            positions=np.array([[2.0, 0.0, 0.0], [3.0, 0.0, 0.0]]),
            sample_times=np.array([0.0, 0.5, 1.0, 1.5]),
        ),
        field=sensors_field,
    )
    np.testing.assert_allclose(
        sensors.get_truth(), [[[22.0, 22.25, 23.0, np.nan]], [[np.nan] * 4]]
    )
    np.testing.assert_allclose(
        sampler.sample(np.array([[0.0, 0.0, 0.0]])), [[[20.0, 21.0]]]
    )
    assert np.isnan(sampler.sample(np.array([[3.0, 0.0, 0.0]]))).all()
    assert sensors.get_descriptor() == sens.DescriptorFactory.scalar()

    descriptor = sens.DescriptorFactory.temperature()
    labelled = sens.SensorsPoint(sensors.get_sensor_data(), sensors_field, descriptor)
    assert labelled.get_descriptor() is descriptor


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


def test_analytic_vector_and_tensor_rotations() -> None:
    data = sens.SensorData(
        positions=np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]]),
        angles=(Rotation.identity(), Rotation.from_euler("z", 90, degrees=True)),
    )
    times = np.array([0.0])
    vector_field = sens.FieldVector(
        sens.FieldSampleAnalytic(
            evaluate=evaluate_vector, default_times=times, num_components=2
        ),
        comp_keys=("x", "y"),
        spatial_dims=sens.EDim.TWOD,
    )
    vector = sens.SensorsPoint(
        sensor_data=data,
        field=vector_field,
    )
    tensor_field = sens.FieldTensor(
        sens.FieldSampleAnalytic(
            evaluate=evaluate_tensor, default_times=times, num_components=3
        ),
        norm_comp_keys=("xx", "yy"),
        dev_comp_keys=("xy",),
        spatial_dims=sens.EDim.TWOD,
    )
    tensor = sens.SensorsPoint(
        sensor_data=data,
        field=tensor_field,
    )
    np.testing.assert_allclose(vector.get_truth()[..., 0], [[2.0, 3.0], [3.0, -2.0]])
    np.testing.assert_allclose(
        tensor.get_truth()[..., 0], [[2.0, 3.0, 1.0], [3.0, 2.0, -1.0]]
    )
    assert vector.get_descriptor() == sens.DescriptorFactory.vector()
    assert tensor.get_descriptor() == sens.DescriptorFactory.tensor(sens.EDim.TWOD)


def test_replacing_sampler_clears_caches_and_refreshes_time_shape() -> None:
    sim_data, _ = scalar_linear_2d()
    positions = np.array([[1.0, 1.0, 0.0]])
    sensors_field = sens.FieldScalar(
        sim_data,
        comp_key="temperature",
        spatial_dims=sens.EDim.TWOD,
    )
    sensors = sens.SensorsPoint(
        sensor_data=sens.SensorData(positions=positions),
        field=sensors_field,
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


def evaluate_shifted_temperature(
    points: npt.NDArray[np.float64], times: npt.NDArray[np.float64]
) -> npt.NDArray[np.float64]:
    return 2.0 * evaluate_temperature(points, times)


def test_sampler_swap_is_local_and_rebinds_field_errors() -> None:
    times = np.array([0.0, 1.0])
    original = sens.FieldSampleAnalytic(
        evaluate=evaluate_temperature, default_times=times, num_components=1
    )
    field = sens.FieldScalar(original, "temperature", sens.EDim.TWOD)
    data = sens.SensorData(positions=np.array([[1.0, 0.0, 0.0]]))
    first = sens.SensorsPoint(data, field)
    second = sens.SensorsPoint(data, field)
    first.set_error_chain(
        [
            sens.ErrSysField(
                field, sens.ErrFieldData(pos_offset_xyz=np.array([[0.5, 0.0, 0.0]]))
            )
        ]
    )

    np.testing.assert_allclose(first.sim_measurements(), [[[21.5, 22.5]]])
    np.testing.assert_allclose(second.get_truth(), [[[21.0, 22.0]]])
    replacement = sens.FieldSampleAnalytic(
        evaluate=evaluate_shifted_temperature, default_times=times, num_components=1
    )

    detached = first.get_field().with_sampler(replacement)
    np.testing.assert_allclose(detached.sample_field(data.positions), [[[42.0, 44.0]]])
    assert first.get_field() is field
    first.set_field_sampler(replacement)

    assert first.get_field() is not field
    assert second.get_field() is field
    np.testing.assert_allclose(second.get_truth(), [[[21.0, 22.0]]])
    np.testing.assert_allclose(first.get_truth(), [[[42.0, 44.0]]])
    np.testing.assert_allclose(first.sim_measurements(), [[[43.0, 45.0]]])
    np.testing.assert_allclose(first.get_errors_total(), [[[1.0, 1.0]]])

    cached = first.get_measurements()
    with pytest.raises(ValueError, match="one-dimensional"):
        first.set_sensor_data(
            sens.SensorData(positions=data.positions, sample_times=np.zeros((2, 1)))
        )
    assert first.get_measurements() is cached

    first.set_error_chain([sens.ErrRandGen(sens.GenNormal(std=1.0, seed=123))])
    first.sim_measurements()
    control = first.copy_for_simulation()

    first.set_field_sampler(original)
    first.sim_measurements()
    control.sim_measurements()
    np.testing.assert_array_equal(first.get_errors_random(), control.get_errors_random())
