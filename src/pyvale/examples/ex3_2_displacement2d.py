# ================================================================================
# pyvale: the python validation engine
# License: MIT
# Copyright (C) 2025 The Computer Aided Validation Team
# ================================================================================

from pathlib import Path
import numpy as np
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

    descriptor = pyv.SensorDescriptor()
    descriptor.name = 'Disp.'
    descriptor.symbol = r'u'
    descriptor.units = r'm'
    descriptor.tag = 'DS'
    descriptor.components = ('x','y','z')

    spat_dims = 2
    field_key = 'disp'
    components = ('disp_x','disp_y')
    disp_field = pyv.FieldVector(sim_data,field_key,components,spat_dims)

    n_sens = (2,3,1)
    x_lims = (0.0,100.0)
    y_lims = (0.0,150.0)
    z_lims = (0.0,0.0)
    sens_pos = pyv.create_sensor_pos_array(n_sens,x_lims,y_lims,z_lims)

    use_sim_time = False
    if use_sim_time:
        sample_times = None
    else:
        sample_times = np.linspace(0.0,np.max(sim_data.time),50)

    sens_data = pyv.SensorData(positions=sens_pos,
                                  sample_times=sample_times)

    disp_sens_array = pyv.SensorArrayPoint(sens_data,
                                              disp_field,
                                              descriptor)

    error_chain = []
    error_chain.append(pyv.ErrSysUniform(low=-0.01e-3,high=0.01e-3))
    error_chain.append(pyv.ErrRandNormal(std=0.01e-3))
    error_int = pyv.ErrIntegrator(error_chain,
                                       sens_data,
                                       disp_sens_array.get_measurement_shape())
    disp_sens_array.set_error_integrator(error_int)


    plot_field = 'disp_x'
    if plot_field == 'disp_x':
        pv_plot = pyv.plot_point_sensors_on_sim(disp_sens_array,'disp_x')
        pv_plot.show(cpos="xy")
    elif plot_field == 'disp_y':
        pv_plot = pyv.plot_point_sensors_on_sim(disp_sens_array,'disp_y')
        pv_plot.show(cpos="xy")

    pyv.plot_time_traces(disp_sens_array,'disp_x')
    pyv.plot_time_traces(disp_sens_array,'disp_y')
    plt.show()


if __name__ == "__main__":
    main()