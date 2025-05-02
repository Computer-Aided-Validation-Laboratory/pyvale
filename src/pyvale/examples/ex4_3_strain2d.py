# ================================================================================
# pyvale: the python validation engine
# License: MIT
# Copyright (C) 2025 The Computer Aided Validation Team
# ================================================================================

import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation as R
import mooseherder as mh
import pyvale as pyv

def main() -> None:
    """pyvale example: strain sensors on a 2D plate with a hole
    ----------------------------------------------------------------------------
    - Demonstrates rotation of tensor fields
    """
    data_path = pyv.DataSet.mechanical_2d_path()
    sim_data = mh.ExodusReader(data_path).read_all_sim_data()
    # Scale to mm to make 3D visualisation scaling easier
    sim_data.coords = sim_data.coords*1000.0 # type: ignore

    descriptor = pyv.SensorDescriptorFactory.strain_descriptor()

    spat_dims = 2
    field_key = 'strain'
    norm_components = ('strain_xx','strain_yy')
    dev_components = ('strain_xy',)
    strain_field = pyv.FieldTensor(sim_data,
                                    field_key,
                                    norm_components,
                                    dev_components,
                                    spat_dims)

    n_sens = (2,3,1)
    x_lims = (0.0,100.0)
    y_lims = (0.0,150.0)
    z_lims = (0.0,0.0)
    sensor_positions = pyv.create_sensor_pos_array(n_sens,x_lims,y_lims,z_lims)

    use_sim_time = False
    if use_sim_time:
        sample_times = None
    else:
        sample_times = np.linspace(0.0,np.max(sim_data.time),50)

    sens_data_norot = pyv.SensorData(positions=sensor_positions,
                                        sample_times=sample_times)

    sg_array_norot = pyv.SensorArrayPoint(sens_data_norot,
                                             strain_field,
                                             descriptor)

    meas_norot = sg_array_norot.get_measurements()

    sens_angles = sensor_positions.shape[0] * \
        (R.from_euler("zyx", [45, 0, 0], degrees=True),)

    sens_data_rot = pyv.SensorData(positions=sensor_positions,
                                      sample_times=sample_times,
                                      angles=sens_angles)

    sg_array_rot = pyv.SensorArrayPoint(sens_data_rot,
                                           strain_field,
                                           descriptor)

    angle_offset = np.zeros_like(sensor_positions)
    angle_offset[:,0] = 1.0 # only rotate about z in 2D
    angle_error_data = pyv.ErrFieldData(ang_offset_zyx=angle_offset)

    sys_err_rot = pyv.ErrSysField(strain_field,angle_error_data)
    err_int = pyv.ErrIntegrator([sys_err_rot],
                                     sens_data_rot,
                                     sg_array_rot.get_measurement_shape())
    sg_array_rot.set_error_integrator(err_int)


    meas_rot = sg_array_rot.get_measurements()


    print(80*'-')
    sens_num = 4
    print('The last 5 time steps (measurements) of sensor {sens_num}:')
    pyv.print_measurements(sg_array_rot,
                              (sens_num-1,sens_num),
                              (1,2),
                              (meas_rot.shape[2]-5,meas_rot.shape[2]))
    print(80*'-')

    plot_comp = 'strain_yy'
    pyv.plot_time_traces(sg_array_norot,plot_comp)
    pyv.plot_time_traces(sg_array_rot,plot_comp)
    plt.show()


if __name__ == "__main__":
    main()