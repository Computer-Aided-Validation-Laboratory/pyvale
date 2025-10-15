import pytest
import pytest_mock
import pyvale.sensorsim as sens
import matplotlib.pyplot as plt

import pyvale.mooseherder as mh
import pyvale.dataset as dataset
from pyvale.sensorsim.visualopts import TraceOptsSensor


@pytest.fixture
def make_data():

    data_path = dataset.thermal_2d_path()
    sim_data = mh.ExodusReader(data_path).read_all_sim_data()

    sim_data = sens.scale_length_units(scale=1000.0,
                                    sim_data=sim_data,
                                    disp_comps=None)

    n_sens = (3,2,1)
    x_lims = (0.0,100.0)
    y_lims = (0.0,50.0)
    z_lims = (0.0,0.0)
    sens_pos = sens.create_sensor_pos_array(n_sens,x_lims,y_lims,z_lims)

    sens_data = sens.SensorData(positions=sens_pos)

    field_key: str = "temperature"
    tc_array = sens.SensorArrayFactory \
        .thermocouples_basic_errs(sim_data,
                                sens_data,
                                elem_dims=2,
                                field_name=field_key)
    
    measurements = tc_array.calc_measurements()

    return tc_array, field_key


def test_fixture(make_data):

    assert make_data[0] != None

testdata = [(2, 4),
            (6, 1),
            (None, 1)
            ]

@pytest.mark.parametrize("sensors_per_plot, expected", testdata)
def test_subplot_made(make_data, sensors_per_plot, expected):

    num_sens = make_data[0]._sensor_data.positions.shape[0]
    if sensors_per_plot == None:
        sensors_per_plot = num_sens
    trace_opts_class = TraceOptsSensor(sensors_per_plot=sensors_per_plot)
    (fig,ax) = sens.plot_time_traces(make_data[0],make_data[1], trace_opts=trace_opts_class)

    assert len(ax) == expected

