'''
================================================================================
example: displacement sensors on a 2d plate

pyvale: the python validation engine
License: MIT
Copyright (C) 2025 The Computer Aided Validation Team
================================================================================
'''
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import mooseherder as mh
import pyvale

def main() -> None:
    data_path = Path('src/data/case17_out.e')
    sim_data = mh.ExodusReader(data_path).read_all_sim_data()
    # Scale m to mm to make 3D visualisation scaling correct for pyvista
    sim_data = pyv.scale_length_units(1000.0,sim_data)

    descriptor = pyvale.SensorDescriptor()
    descriptor.name = 'Displacement'
    descriptor.symbol = r'u'
    descriptor.units = r'm'
    descriptor.tag = 'DS'
    descriptor.components = ('x','y','z')

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

    pos_gen = pyvale.GeneratorNormal(std = 1.0)
    angle_gen = pyvale.GeneratorNormal(std = 1.0)
    time_gen = pyvale.GeneratorNormal(std = 0.1)

    pos_offset_xyz = 1.5*np.ones_like(sensor_positions)
    pos_offset_xyz[:,2] = 0 # don't perturb z in 2D

    ang_offset_zyx = np.zeros_like(sensor_positions)
    ang_offset_zyx[:,0] = 1.5 # only rotate about z in 2D


    field_err_data = pyvale.ErrFieldData(
        pos_offset_xyz = pos_offset_xyz,
        ang_offset_zyx = ang_offset_zyx,
        time_offset = 1.0,
        pos_rand_xyz = (pos_gen,pos_gen,None),
        ang_rand_zyx = (angle_gen,None,None),
        time_rand = time_gen
    )
    sys_err_field = pyvale.ErrSysField(disp_field,
                                       sensor_data,
                                       field_err_data)

    error_int = pyvale.ErrIntegrator([sys_err_field],
                                       disp_sens_array.get_measurement_shape())
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

    sens_data_by_chain = error_int.get_sens_data_by_chain()
    for ii,ss in enumerate(sens_data_by_chain):
        if ss is not None:
            print(f"SensorData @ [{ii}]")
            print(ss)

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