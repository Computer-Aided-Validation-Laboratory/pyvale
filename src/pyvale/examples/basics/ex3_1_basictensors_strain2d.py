# ==============================================================================
# pyvale: the python validation engine
# License: MIT
# Copyright (C) 2025 The Computer Aided Validation Team
# ==============================================================================


import numpy as np
import matplotlib.pyplot as plt
import mooseherder as mh
import pyvale as pyv

def main() -> None:

    data_path = pyv.DataSet.mechanical_2d_path()
    sim_data = mh.ExodusReader(data_path).read_all_sim_data()
    sim_data = pyv.scale_length_units(scale=1000.0,
                                      sim_data=sim_data,
                                      disp_comps=("disp_x","disp_y"))

    n_sens = (2,3,1)
    x_lims = (0.0,100.0)
    y_lims = (0.0,150.0)
    z_lims = (0.0,0.0)
    sens_pos = pyv.create_sensor_pos_array(n_sens,x_lims,y_lims,z_lims)

    sample_times = np.linspace(0.0,np.max(sim_data.time),50)

    sens_data = pyv.SensorData(positions=sens_pos,
                                  sample_times=sample_times)

    straingauge_array = pyv.SensorArrayFactory \
                            .strain_gauges_basic_errs(sim_data,
                                                      sens_data,
                                                      "strain",
                                                      elem_dims=2)

    plot_field = "strain_yy"
    pv_plot = pyv.plot_point_sensors_on_sim(straingauge_array,plot_field)
    pv_plot.show(cpos="xy")

    pyv.plot_time_traces(straingauge_array,"strain_xx")
    pyv.plot_time_traces(straingauge_array,"strain_yy")
    pyv.plot_time_traces(straingauge_array,"strain_xy")
    plt.show()


if __name__ == "__main__":
    main()