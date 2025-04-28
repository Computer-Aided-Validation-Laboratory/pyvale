# ================================================================================
# pyvale: the python validation engine
# License: MIT
# Copyright (C) 2025 The Computer Aided Validation Team
# ================================================================================

import matplotlib.pyplot as plt
import mooseherder as mh
import pyvale as pyv

def main() -> None:
    """pyvale example: displacement sensors on a 2D plate with a hole
    ----------------------------------------------------------------------------
    """
    data_path = pyv.DataSet.mechanical_2d_path()
    sim_data = mh.ExodusReader(data_path).read_all_sim_data()
    # Scale to mm to make 3D visualisation scaling easier
    sim_data.coords = sim_data.coords*1000.0 # type: ignore

    n_sens = (2,3,1)
    x_lims = (0.0,100.0)
    y_lims = (0.0,150.0)
    z_lims = (0.0,0.0)
    sens_pos = pyv.create_sensor_pos_array(n_sens,x_lims,y_lims,z_lims)

    sens_data = pyv.SensorData(positions=sens_pos)

    disp_sens_array = pyv.SensorArrayFactory \
                            .disp_sensors_basic_errs(sim_data,
                                                     sens_data,
                                                     "displacement",
                                                     spat_dims=2)

    plot_field = 'disp_x'
    pv_plot = pyv.plot_point_sensors_on_sim(disp_sens_array,plot_field)
    pv_plot.show(cpos="xy")

    pyv.plot_time_traces(disp_sens_array,'disp_x')
    pyv.plot_time_traces(disp_sens_array,'disp_y')
    plt.show()


if __name__ == "__main__":
    main()