# ==============================================================================
# pyvale: the python validation engine
# License: MIT
# Copyright (C) 2025 The Computer Aided Validation Team
# ==============================================================================
import copy
import matplotlib.pyplot as plt
import numpy as np
import pyvale.sensorsim as sens
import pyvale.verif as verif


def main() -> None:
    # 10x7.5 plate with bi-directional field gradient
    # 40x30 elements [x,y]
    # Field slope of 20/lengX in X
    # Field slope of 10/lengY in Y
    # Field max in top corner of 220, field min in bottom corner 20

    (sim_data,_) = verif.analyticsimdatafactory.scalar_linear_2d()
    sim_data_nomesh = copy.deepcopy(sim_data)

    # When we have a point cloud and no mesh the connectivity table is None
    sim_data_nomesh.connect = None

    descriptor = sens.SensorDescriptorFactory.temperature_descriptor()

    field_key = 'temperature'
    scal_field = sens.FieldScalar(sim_data,
                                  field_key=field_key,
                                  elem_dims=2)

    scal_field_nm = sens.FieldScalar(sim_data_nomesh,
                                     field_key=field_key,
                                     elem_dims=2)


    n_sens = (4,1,1)
    x_lims = (0.0,10.0)
    y_lims = (0.0,7.5)
    z_lims = (0.0,0.0)
    sens_pos = sens.gen_pos_grid_inside(n_sens,x_lims,y_lims,z_lims)

    use_sim_time = False
    if use_sim_time:
        sample_times = None
    else:
        sample_times = np.linspace(0.0,np.max(sim_data.time),50)

    sensor_data = sens.SensorData(positions=sens_pos,
                                 sample_times=sample_times)

    tc_array = sens.SensorsPoint(sensor_data,
                                    scal_field,
                                    descriptor)

    tc_array_nm = sens.SensorsPoint(sensor_data,
                                        scal_field_nm,
                                        descriptor)


    meas = tc_array.get_measurements()
    meas_nm = tc_array_nm.get_measurements()

    print(80*"-")
    print("MESH INTERP")
    sens.print_measurements(tc_array,
                            slice(0,1), # Sensor 1
                            slice(0,1), # Component 1: scalar field = 1 component
                            slice (meas.shape[2]-5,meas.shape[2]))

    print("POINT INTERP")
    sens.print_measurements(tc_array_nm,
                            slice(0,1), # Sensor 1
                            slice(0,1), # Component 1: scalar field = 1 component
                            slice (meas_nm.shape[2]-5,meas_nm.shape[2]))

    print(f"{np.allclose(meas,meas_nm)=}")
    print(80*"-")

    # (fig,ax) = sens.plot_time_traces(tc_array,field_key)
    # ax.set_title("Mesh Interp.")
    # plt.show()

    pv_plot = sens.plot_point_sensors_on_sim(tc_array_nm,field_key)
    pv_plot.show(cpos="xy")

if __name__ == '__main__':
    main()
