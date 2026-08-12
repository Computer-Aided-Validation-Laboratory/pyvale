import pytest
#import pytest_mock
import pyvale.sensorsim as sens
import matplotlib.pyplot as plt

import pyvale.mooseherder as mh
import pyvale.dataset as dataset
from pyvale.sensorsim.visualopts import TraceOptsSensor

from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation
import matplotlib.pyplot as plt


@pytest.fixture
def make_therm_data() -> tuple[sens.ISensorArray,str]:

    data_path = dataset.thermal_2d_path()
    sim_data = mh.ExodusLoader(data_path).load_all_sim_data()

    sim_data = sens.scale_length_units(scale=1000.0,
                                        sim_data=sim_data,
                                        disp_keys=None)

    n_sens = (3,2,1)
    x_lims = (0.0,100.0)
    y_lims = (0.0,50.0)
    z_lims = (0.0,0.0)
    sens_pos = sens.gen_pos_grid_inside(n_sens,x_lims,y_lims,z_lims)


    sens_data = sens.SensorData(positions=sens_pos)
    field_key: str = "temperature"

    sens_array = sens.SensorFactory.scalar_point(
        sim_data,
        sens_data,
        comp_key=field_key,
        spatial_dims=sens.EDim.TWOD,
        descriptor=sens.DescriptorFactory.temperature(),
    )

    err_chain= [
        sens.ErrSysGen(sens.GenUniform(low=-5.0,high=5.0)),
        sens.ErrRandGen(sens.GenNormal(std=2.0)),
    ]

    sens_array.set_error_chain(err_chain)


    return sens_array, field_key

@pytest.fixture
def make_disp_data() -> tuple[sens.ISensorArray,str]:

    data_path2: Path = dataset.mechanical_2d_path()
    sim_data2: io.SimData = mh.ExodusLoader(data_path2).load_all_sim_data()

    disp_keys = ("disp_x","disp_y")
    strain_norm_keys = ("strain_xx","strain_yy",)
    strain_dev_keys = ("strain_xy",)

    sim_data2: io.SimData  = sens.scale_length_units(scale=1000.0,
                                                    sim_data=sim_data2,
                                                    disp_keys=disp_keys)

    sens_pos2: np.ndarray = sens.gen_pos_grid_inside(num_sensors=(2,2,1),
                                                    x_lims=(0.0,100.0),
                                                    y_lims=(0.0,150.0),
                                                    z_lims=(0.0,0.0))

    sample_times2: np.ndarray = np.linspace(0.0,np.max(sim_data2.time),50)

    sens_angles2: tuple[Rotation] = sens_pos2.shape[0] * \
        (Rotation.from_euler("zyx",[90,0,0], degrees=True),)

    disp_sens_data2 = sens.SensorData(positions=sens_pos2,
                                    sample_times=sample_times2,
                                    angles=sens_angles2)

    disp_sens: sens.SensorsPoint = sens.SensorFactory.vector_point(
        sim_data2,
        disp_sens_data2,
        comp_keys=disp_keys,
        spatial_dims=sens.EDim.TWOD,
        descriptor=sens.DescriptorFactory.displacement(),
)

    return disp_sens, disp_keys


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

def test_pyvista_subplots_made(make_therm_data, make_disp_data):
    pv_plot = sens.plot_point_sensors_on_sim([make_therm_data[0], make_disp_data[0]],
                                             [make_therm_data[1], make_disp_data[1][0]])
    assert pv_plot.shape == (1,2)
