# ==============================================================================
# pyvale: the python validation engine
# License: MIT
# Copyright (C) 2025 The Computer Aided Validation Team
# ==============================================================================
import copy
import matplotlib.pyplot as plt
import numpy as np
import pyvale.sensorsim as sens
import pyvale.verif as va

def main() -> None:
    # 10x7.5 plate with bi-directional field gradient
    # 40x30 elements [x,y]
    # Field slope of 20/lengX in X
    # Field slope of 10/lengY in Y
    # Field max in top corner of 220, field min in bottom corner 20
    (sim_data,_) = va.AnalyticCaseFactory.scalar_linear_2d()
    sim_data_nomesh = copy.deepcopy(sim_data)
    sim_data_nomesh.connect = None

    descriptor = sens.SensorDescriptorFactory.temperature_descriptor()

    field_key = 'scalar'
    scal_field = sens.FieldScalar(sim_data,
                                  field_key=field_key,
                                  elem_dims=2)

    # scal_field_nm = sens.FieldScalar(sim_data_nomesh,
    #                                  field_key=field_key,
    #                                  elem_dims=2)


    n_sens = (4,1,1)
    x_lims = (0.0,10.0)
    y_lims = (0.0,7.5)
    z_lims = (0.0,0.0)
    sens_pos = sens.create_sensor_pos_array(n_sens,x_lims,y_lims,z_lims)

    use_sim_time = False
    if use_sim_time:
        sample_times = None
    else:
        sample_times = np.linspace(0.0,np.max(sim_data.time),50)

    sensor_data = sens.SensorData(positions=sens_pos,
                                 sample_times=sample_times)

    tc_array = sens.SensorArrayPoint(sensor_data,
                                    scal_field,
                                    descriptor)

    measurements = tc_array.get_measurements()

    sens.print_measurements(tc_array,
                            slice(0,1), # Sensor 1
                            slice(0,1), # Component 1: scalar field = 1 component
                            slice (measurements.shape[2]-5,measurements.shape[2]))

    # (fig,_) = sens.plot_time_traces(tc_array,field_key)
    # plt.show()

    # pv_plot = sens.plot_point_sensors_on_sim(tc_array,field_key)
    # pv_plot.show(cpos="xy")

if __name__ == '__main__':
    main()
