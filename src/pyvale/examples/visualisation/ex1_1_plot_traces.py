# ==============================================================================
# pyvale: the python validation engine
# License: MIT
# Copyright (C) 2025 The Computer Aided Validation Team
# ==============================================================================

"""
Pyvale example: TODO
--------------------------------------------------------------------------------
TODO

Test case: TODO
"""

import numpy as np
import matplotlib.pyplot as plt
import mooseherder as mh
import pyvale as pyv

# TODO: comments and full description for this example like the basics examples

def main() -> None:

    data_path = pyv.DataSet.thermal_2d_path()
    sim_data = mh.ExodusReader(data_path).read_all_sim_data()
    sim_data = pyv.scale_length_units(scale=1000.0,
                                      sim_data=sim_data,
                                      disp_comps=None)

    n_sens = (4,1,1)
    x_lims = (0.0,100.0)
    y_lims = (0.0,50.0)
    z_lims = (0.0,0.0)
    sens_pos = pyv.create_sensor_pos_array(n_sens,x_lims,y_lims,z_lims)

    sample_times = np.linspace(0.0,np.max(sim_data.time),12)

    sens_data = pyv.SensorData(positions=sens_pos,
                                  sample_times=sample_times)

    field_key = "temperature"
    tc_array = pyv.SensorArrayFactory \
        .thermocouples_basic_errs(sim_data,
                                  sens_data,
                                  field_key,
                                  elem_dims=2)

    err_int = pyv.ErrIntegrator([pyv.ErrSysOffset(offset=-5.0)],
                                     sens_data,
                                     tc_array.get_measurement_shape())
    tc_array.set_error_integrator(err_int)

    measurements = tc_array.get_measurements()


    print(80*"-")

    sens_print: int = 0
    time_print: int = 5
    comp_print: int = 0

    print(f"These are the last {time_print} virtual measurements of sensor "
          + f"{sens_print}:")

    pyv.print_measurements(sens_array=tc_array,
                           sensors=(sens_print,sens_print+1),
                           components=(comp_print,comp_print+1),
                           time_steps=(measurements.shape[2]-time_print,
                                       measurements.shape[2]))
    print(80*"-")


    trace_props = pyv.TraceOptsSensor()

    trace_props.truth_line = None
    trace_props.sim_line = None
    pyv.plot_time_traces(tc_array,field_key,trace_props)

    trace_props.meas_line = "--o"
    trace_props.truth_line = "-x"
    trace_props.sim_line = ":+"
    pyv.plot_time_traces(tc_array,field_key,trace_props)

    trace_props.sensors_to_plot = np.arange(measurements.shape[0]-2
                                           ,measurements.shape[0])
    pyv.plot_time_traces(tc_array,field_key,trace_props)

    trace_props.sensors_to_plot = None
    trace_props.time_min_max = (0.0,100.0)
    pyv.plot_time_traces(tc_array,field_key,trace_props)

    plt.show()

    pv_plot = pyv.plot_point_sensors_on_sim(tc_array,field_key)
    pv_plot.camera_position = [(-7.547, 59.753, 134.52),
                                   (41.916, 25.303, 9.297),
                                   (0.0810, 0.969, -0.234)]
    pv_plot.show()


if __name__ == "__main__":
    main()
