from collections.abc import Generator
import pytest
import matplotlib.pyplot as plt
import pyvale.sensorsim as sens
import pyvale.mooseherder as mh
import pyvale.data as dataset
from pyvale.sensorsim.visualopts import TraceOptsSensor


@pytest.fixture(autouse=True)
def cleanup_matplotlib() -> Generator[None, None, None]:
    """Closes all matplotlib figures after each test to prevent memory leaks."""
    yield
    plt.close("all")


@pytest.fixture
def make_data() -> tuple[sens.ISensorArray, str]:
    data_path = dataset.thermal_2d_path()
    sim_data = mh.ExodusLoader(data_path).load_all_sim_data()

    sim_data = sens.scale_length_units(
        scale=1000.0, sim_data=sim_data, disp_keys=None
    )

    n_sens = (3, 2, 1)
    x_lims = (0.0, 100.0)
    y_lims = (0.0, 50.0)
    z_lims = (0.0, 0.0)
    sens_pos = sens.gen_pos_grid_inside(n_sens, x_lims, y_lims, z_lims)

    sens_data = sens.SensorData(positions=sens_pos)

    field_key = "temperature"
    sens_array = sens.SensorFactory.scalar_point(
        sim_data,
        sens_data,
        comp_key=field_key,
        spatial_dims=sens.EDim.THREED,
        descriptor=sens.DescriptorFactory.temperature(),
    )

    err_chain = [
        sens.ErrSysGenPercent(sens.GenUniform(low=-1.0, high=1.0)),
        sens.ErrRandGenPercent(sens.GenNormal(std=1.0)),
    ]
    sens_array.set_error_chain(err_chain)
    _ = sens_array.sim_measurements()

    return sens_array, field_key


def test_fixture(make_data: tuple[sens.ISensorArray, str]) -> None:
    assert make_data[0] is not None


testdata = [
    (2, 4),
    (6, 1),
    (None, 1),
    (7, 1),
]


@pytest.mark.parametrize("sensors_per_plot, expected", testdata)
def test_subplot_made(
    make_data: tuple[sens.ISensorArray, str],
    sensors_per_plot: int | None,
    expected: int,
) -> None:
    num_sens = make_data[0].get_sensor_data().positions.shape[0]
    if sensors_per_plot is None:
        sensors_per_plot = num_sens
    trace_opts_class = TraceOptsSensor(sensors_per_plot=sensors_per_plot)
    (fig, ax) = sens.plot_time_traces(
        make_data[0], make_data[1], trace_opts=trace_opts_class
    )
    try:
        assert len(ax) == expected
    finally:
        plt.close(fig)

