'''
================================================================================
example: displacement sensors on a 2d plate

pyvale: the python validation engine
License: MIT
Copyright (C) 2024 The Computer Aided Validation Team
================================================================================
'''
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import mooseherder as mh
import pyvale

def main() -> None:
    data_path = Path('src/data/case17_out.e')
    data_reader = mh.ExodusReader(data_path)
    sim_data = data_reader.read_all_sim_data()
    # Scale to mm to make 3D visualisation scaling easier
    sim_data.coords = sim_data.coords*1000.0 # type: ignore

    descriptor = pyvale.SensorDescriptorFactory.displacement_descriptor()

    spat_dims = 2
    field_key = 'disp'
    components = ('disp_x','disp_y')
    disp_field = pyvale.FieldVector(sim_data,field_key,components,spat_dims)

    n_sens = (2,3,1)
    x_lims = (0.0,100.0)
    y_lims = (0.0,150.0)
    z_lims = (0.0,0.0)
    sensor_positions = pyvale.create_sensor_pos_array(n_sens,x_lims,y_lims,z_lims)

    use_sim_time = True
    if use_sim_time:
        sample_times = None
    else:
        sample_times = np.linspace(0.0,np.max(sim_data.time),50)

    sensor_data = pyvale.SensorData(positions=sensor_positions,
                                    sample_times=sample_times)

    disp_sens_array = pyvale.SensorArrayPoint(sensor_data,
                                              disp_field,
                                              descriptor)



    angle_offset = np.zeros_like(sensor_positions)
    angle_offset[:,0] = 45.0 # only rotate about z in 2D

    field_error_data = pyvale.ErrFieldData(ang_offset_zyx=angle_offset,
                                             spatial_averager=pyvale.EIntSpatialType.RECT4PT,
                                             spatial_dims=np.array([2.0,2.0,0.0]))

    field_errs = []
    field_errs.append(pyvale.ErrSysField(disp_field,
                                        field_error_data))

    err_int_opts = pyvale.ErrIntOpts(force_dependence=True,
                                               store_errs_by_func=True)
    error_int = pyvale.ErrIntegrator(field_errs,
                                       sensor_data,
                                       disp_sens_array.get_measurement_shape(),
                                       err_int_opts)
    disp_sens_array.set_error_integrator(error_int)

    measurements = disp_sens_array.calc_measurements()

    print(80*'-')
    sens_num = 4
    print('The last 5 time steps (measurements) of sensor {sens_num}:')
    pyvale.print_measurements(disp_sens_array,
                              (sens_num-1,sens_num),
                              (0,1),
                              (measurements.shape[2]-5,measurements.shape[2]))
    print(80*'-')

    plot_field = 'disp_x'
    if plot_field == 'disp_x':
        pv_plot = pyvale.plot_point_sensors_on_sim(disp_sens_array,'disp_x')
        pv_plot.show(cpos="xy")
    elif plot_field == 'disp_y':
        pv_plot = pyvale.plot_point_sensors_on_sim(disp_sens_array,'disp_y')
        pv_plot.show(cpos="xy")

    pyvale.plot_time_traces(disp_sens_array,'disp_x')
    pyvale.plot_time_traces(disp_sens_array,'disp_y')
    plt.show()


if __name__ == "__main__":
    main()