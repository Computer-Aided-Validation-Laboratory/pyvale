# ==============================================================================
# pyvale: the python validation engine
# License: MIT
# Copyright (C) 2025 The Computer Aided Validation Team
# ==============================================================================

"""
Pyvale example:
----------------------------------------------------------------------------
"""

import numpy as np
import matplotlib.pyplot as plt
import mooseherder as mh
import pyvale as pyv


def main() -> None:


    data_path = pyv.DataSet.thermomechanical_2d_path()
    sim_data = mh.ExodusReader(data_path).read_all_sim_data()
    # Scale to mm to make 3D visualisation scaling easier as pyvista scales
    # everything to unity
    sim_data = pyv.scale_length_units(scale=1000.0,
                                      sim_data=sim_data,
                                      disp_comps=("disp_x","disp_y","disp_z"))


    n_sens = (4,1,1)
    x_lims = (0.0,100.0)
    y_lims = (0.0,50.0)
    z_lims = (0.0,0.0)
    sens_pos = pyv.create_sensor_pos_array(n_sens,x_lims,y_lims,z_lims)


    sample_times = np.linspace(0.0,np.max(sim_data.time),50)

    sens_data = pyv.SensorData(positions=sens_pos,
                               sample_times=sample_times)


    tc_field = 'temperature'
    tc_array = pyv.SensorArrayFactory \
        .thermocouples_basic_errs(sim_data,
                                  sens_data,
                                  tc_field,
                                  elem_dims=2)

    sg_field = 'strain'
    sg_array = pyv.SensorArrayFactory \
        .strain_gauges_basic_errs(sim_data,
                                  sens_data,
                                  sg_field,
                                  elem_dims=2)

    #===========================================================================
    # Visualise Traces
    print(80*'-')
    sens_num = 4
    print('THERMAL: The last 5 time steps (measurements) of sensor {sens_num}:')
    pyv.print_measurements(tc_array,
                              (sens_num-1,sens_num),
                              (0,1),
                              (tc_array.get_measurement_shape()[2]-5,
                               tc_array.get_measurement_shape()[2]))
    print(80*'-')

    pyv.plot_time_traces(tc_array,"temperature")
    pyv.plot_time_traces(sg_array,"strain_xx")
    plt.show()


if __name__ == "__main__":
    main()