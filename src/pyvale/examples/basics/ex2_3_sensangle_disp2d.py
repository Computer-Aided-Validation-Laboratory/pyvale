# ==============================================================================
# pyvale: the python validation engine
# License: MIT
# Copyright (C) 2025 The Computer Aided Validation Team
# ==============================================================================

"""
Pyvale example: Sensor angles for vector fields
----------------------------------------------------------------------------
In this example we demonstrate how to setup vector field sensors at custom
orientations with respect to the simulation coordinate system. We first build a
sensor array aligned with the simulation coords in the same way as the previous
example. We then build a sensor array with the sensors rotated and compare this
to the case with no rotation.

Note that this tutorial assumes you are familiar with the use of pyvale for
scalar fields as described in the first set of examples.

Test case: point displacement sensors on a 2D plate with hole loaded in tension
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation
import mooseherder as mh
import pyvale as pyv

def main() -> None:

    data_path = pyv.DataSet.mechanical_2d_path()
    sim_data = mh.ExodusReader(data_path).read_all_sim_data()

    field_name = "disp"
    field_comps = ("disp_x","disp_y")
    sim_data = pyv.scale_length_units(scale=1000.0,
                                      sim_data=sim_data,
                                      disp_comps=field_comps)

    descriptor = pyv.SensorDescriptorFactory.displacement_descriptor()

    disp_field = pyv.FieldVector(sim_data,field_name,field_comps,elem_dims=2)

    n_sens = (2,3,1)
    x_lims = (0.0,100.0)
    y_lims = (0.0,150.0)
    z_lims = (0.0,0.0)
    sens_pos = pyv.create_sensor_pos_array(n_sens,x_lims,y_lims,z_lims)


    sample_times = np.linspace(0.0,np.max(sim_data.time),50)

    sens_data_norot = pyv.SensorData(positions=sens_pos,
                                     sample_times=sample_times)

    disp_sens_norot = pyv.SensorArrayPoint(sens_data_norot,
                                              disp_field,
                                              descriptor)

    disp_sens_norot.calc_measurements()

    sens_angles = sens_pos.shape[0] * \
        (Rotation.from_euler("zyx", [45, 0, 0], degrees=True),)

    sens_data_rot = pyv.SensorData(positions=sens_pos,
                                      sample_times=sample_times,
                                      angles=sens_angles)

    disp_sens_rot = pyv.SensorArrayPoint(sens_data_rot,
                                            disp_field,
                                            descriptor)


    angle_offset = np.zeros_like(sens_pos)
    angle_offset[:,0] = 1.0 # only rotate about z in 2D
    angle_error_data = pyv.ErrFieldData(ang_offset_zyx=angle_offset)

    sys_err_rot = pyv.ErrSysField(disp_field,angle_error_data)

    sys_err_int = pyv.ErrIntegrator([sys_err_rot],
                                         sens_data_rot,
                                         disp_sens_rot.get_measurement_shape())
    disp_sens_rot.set_error_integrator(sys_err_int)

    meas_rot = disp_sens_rot.get_measurements()


    print(80*'-')
    sens_num = 4
    print('The last 5 time steps (measurements) of sensor {sens_num}:')
    pyv.print_measurements(disp_sens_rot,
                              (sens_num-1,sens_num),
                              (0,1),
                              (meas_rot.shape[2]-5,meas_rot.shape[2]))
    print(80*'-')

    plot_field = 'disp_x'
    if plot_field == 'disp_x':
        pv_plot = pyv.plot_point_sensors_on_sim(disp_sens_rot,'disp_x')
        pv_plot.show(cpos="xy")
    elif plot_field == 'disp_y':
        pv_plot = pyv.plot_point_sensors_on_sim(disp_sens_rot,'disp_y')
        pv_plot.show(cpos="xy")

    (fig,ax) = pyv.plot_time_traces(disp_sens_norot,plot_field)
    (fig,ax) = pyv.plot_time_traces(disp_sens_rot,plot_field)
    plt.show()


if __name__ == "__main__":
    main()