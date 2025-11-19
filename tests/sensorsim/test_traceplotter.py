import pytest
#import pytest_mock
import pyvale.sensorsim as sens
import matplotlib.pyplot as plt

import pyvale.mooseherder as mh
import pyvale.dataset as dataset
from pyvale.sensorsim.visualopts import TraceOptsSensor


@pytest.fixture
def make_therm_data():
    """
    This is a basic set up of thermal data to be plot
    for testing purposes
    """

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

@pytest.fixture
def make_disp_data():
    """
    This is a basic set up of displacement data to be plot
    for testing purposes
    """
    data_path = dataset.mechanical_2d_path()
    sim_data = mh.ExodusReader(data_path).read_all_sim_data()

    field_name = "disp"
    field_comps = ("disp_x","disp_y")
    sim_data = sens.scale_length_units(scale=1000.0,
                                        sim_data=sim_data,
                                        disp_comps=field_comps)

    n_sens = (2,3,1)
    x_lims = (0.0,100.0)
    y_lims = (0.0,150.0)
    z_lims = (0.0,0.0)
    sens_pos = sens.create_sensor_pos_array(n_sens,x_lims,y_lims,z_lims)

    sens_data = sens.SensorData(positions=sens_pos)

    disp_sens_array = sens.SensorArrayFactory \
                        .disp_sensors_basic_errs(sim_data,
                                                    sens_data,
                                                    elem_dims=2,
                                                    field_name=field_name,
                                                    field_comps=field_comps,
                                                    errs_pc=2.0)

    measurements = disp_sens_array.calc_measurements()

    return disp_sens_array, field_comps


def test_fixture_therm(make_therm_data):

    assert make_therm_data[0] != None

def test_fixture_disp(make_disp_data):
    
    assert make_disp_data[0] != None

therm_testdata = [(2, 4),
            (6, 1),
            (None, 1),
            (7, 1)
            ]

@pytest.mark.parametrize("sensors_per_plot, expected", therm_testdata)
def test_mpl_subplot_made(make_therm_data, sensors_per_plot, expected):

    num_sens = make_therm_data[0]._sensor_data.positions.shape[0]
    if sensors_per_plot == None:
        sensors_per_plot = num_sens
    trace_opts_class = TraceOptsSensor(sensors_per_plot=sensors_per_plot)
    (fig,ax) = sens.plot_time_traces(make_therm_data[0],make_therm_data[1], trace_opts=trace_opts_class)

    assert len(ax) == expected

"""
disp_testdata = [([make_therm_data[0], make_disp_data[0]], 
                  [make_therm_data[1], make_disp_data[1][0]], 
                  4),
            ]"""

"""
@pytest.mark.parametrize("things_to_plot, field, expected", disp_testdata)
def test_pyvista_subplot_made(make_therm_data, make_disp_data, things_to_plot, field, expected):

    pv_plot = sens.plot_point_sensors_on_sim(things_to_plot, field)

    assert 1 == 1"""

def test_pyvista_subplots_made(make_therm_data, make_disp_data):
    pv_plot = sens.plot_point_sensors_on_sim([make_therm_data[0], make_disp_data[0]],
                                             [make_therm_data[1], make_disp_data[1][0]])
    assert pv_plot.shape == (1,2)