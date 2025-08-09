# ==============================================================================
# pyvale: the python validation engine
# License: MIT
# Copyright (C) 2025 The Computer Aided Validation Team
# ==============================================================================

import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation
import pyvale.mooseherder as mh
import pyvale as pyv


def main() -> None:

    data_path = pyv.DataSet.mechanical_2d_path()
    sim_data = mh.ExodusLoader(data_path).read_all_sim_data()
<<<<<<< HEAD:src/pyvale/examples/ex3_4_displacement2d.py
    # Scale m to mm to make 3D visualisation scaling correct for pyvista
    sim_data = pyv.scale_length_units(1000.0,sim_data)
=======

    field_name = "disp"
    field_comps = ("disp_x","disp_y")
    sim_data = pyv.scale_length_units(scale=1000.0,
                                      sim_data=sim_data,
                                      disp_comps=field_comps)
>>>>>>> main:dev/old-examples/ex2_X_errangle_disp2d.py

    descriptor = pyv.SensorDescriptorFactory.displacement_descriptor()

    disp_field = pyv.FieldVector(sim_data,field_name,field_comps,elem_dims=2)

    n_sens = (2,2,1)
    x_lims = (0.0,100.0)
    y_lims = (0.0,150.0)
    z_lims = (0.0,0.0)
<<<<<<< HEAD:src/pyvale/examples/ex3_4_displacement2d.py
    sensor_positions = pyv.create_sensor_pos_array(n_sens,
                                                    x_lims,
                                                    y_lims,
                                                    z_lims)
=======
    sens_pos = pyv.create_sensor_pos_array(n_sens,x_lims,y_lims,z_lims)
>>>>>>> main:dev/old-examples/ex2_X_errangle_disp2d.py


    sample_times = np.linspace(0.0,np.max(sim_data.time),50)

    sensor_angles = sens_pos.shape[0] * \
        (Rotation.from_euler("zyx", [0, 0, 0], degrees=True),)

    sensor_data = pyv.SensorData(positions=sens_pos,
                                  sample_times=sample_times,
                                  angles=sensor_angles,
                                  spatial_averager=pyv.EIntSpatialType.QUAD4PT,
                                  spatial_dims=np.array([5.0,5.0,0.0]))

    #---------------------------------------------------------------------------
    disp_sensors = pyv.SensorArrayPoint(sensor_data,
                                           disp_field,
                                           descriptor)

    pos_offset = -10.0*np.ones_like(sens_pos)
    pos_offset[:,2] = 0 # in 2d we only have offset in x and y so zero z

    angle_offset = np.zeros_like(sens_pos)
    angle_offset[:,0] = 5.0 # only rotate about z in 2D

    time_offset = 1.0*np.ones_like(disp_sensors.get_sample_times())

    field_error_data = pyv.ErrFieldData(pos_offset_xyz=pos_offset,
                                           ang_offset_zyx=angle_offset,
                                           time_offset=time_offset)

    error_chain = []
    error_chain.append(pyv.ErrSysField(disp_field,field_error_data))
    error_integrator = pyv.ErrIntegrator(error_chain,
                                            sensor_data,
                                            disp_sensors.get_measurement_shape())

    disp_sensors.set_error_integrator(error_integrator)

    measurements = disp_sensors.calc_measurements()

    #---------------------------------------------------------------------------
    print(80*"-")
    sens_num = 4
    print("The last 5 time steps (measurements) of sensor {sens_num}:")
    pyv.print_measurements(disp_sensors,
                              (sens_num-1,sens_num),
                              (0,1),
                              (measurements.shape[2]-5,measurements.shape[2]))
    print(80*"-")

    #---------------------------------------------------------------------------
    plot_field = "disp_x"

    pv_plot = pyv.plot_point_sensors_on_sim(disp_sensors,plot_field)
    pv_plot.show(cpos="xy")

    pyv.plot_time_traces(disp_sensors,plot_field)
    plt.show()


if __name__ == "__main__":
    main()