import pytest
import pytest_mock
import pyvale.sensorsim as sens
import matplotlib.pyplot as plt

import pyvale.mooseherder as mh
import pyvale.dataset as dataset


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
    #print(f"\nMeasurements for last sensor:\n{measurements[-1,0,:]}\n")

    """

    output_path = Path.cwd() / "pyvale-output"
    if not output_path.is_dir():
        output_path.mkdir(parents=True, exist_ok=True)"""

    (fig,ax) = sens.plot_time_traces(tc_array,field_key)

    return fig, ax



