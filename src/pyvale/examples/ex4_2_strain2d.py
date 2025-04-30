# ================================================================================
# pyvale: the python validation engine
# License: MIT
# Copyright (C) 2025 The Computer Aided Validation Team
# ================================================================================

import numpy as np
import matplotlib.pyplot as plt
import mooseherder as mh
import pyvale as pyv

def main() -> None:
    """pyvale example: strain sensors on a 2D plate with a hole
    ----------------------------------------------------------------------------
    """
    data_path = pyv.DataSet.mechanical_2d_path()
    sim_data = mh.ExodusReader(data_path).read_all_sim_data()
    # Scale to mm to make 3D visualisation scaling easier
    sim_data.coords = sim_data.coords*1000.0 # type: ignore

    descriptor = pyv.SensorDescriptor()
    descriptor.name = 'Strain'
    descriptor.symbol = r'\varepsilon'
    descriptor.units = r'-'
    descriptor.tag = 'SG'
    descriptor.components = ('xx','yy','xy')

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
    sens_pos = pyv.create_sensor_pos_array(n_sens,x_lims,y_lims,z_lims)

    use_sim_time = False
    if use_sim_time:
        sample_times = None
    else:
        sample_times = np.linspace(0.0,np.max(sim_data.time),50)

    sens_data = pyv.SensorData(positions=sens_pos,
                                  sample_times=sample_times)

    straingauge_array = pyv.SensorArrayPoint(sens_data,
                                                strain_field,
                                                descriptor)

    error_chain = []
    error_chain.append(pyv.ErrSysUniform(low=-0.1e-3,high=0.1e-3))
    error_chain.append(pyv.ErrRandNormal(std=0.1e-3))
    error_int = pyv.ErrIntegrator(error_chain,
                                       sens_data,
                                       straingauge_array.get_measurement_shape())
    straingauge_array.set_error_integrator(error_int)

    plot_field = 'strain_yy'
    pv_plot = pyv.plot_point_sensors_on_sim(straingauge_array,plot_field)
    pv_plot.show(cpos="xy")

    pyv.plot_time_traces(straingauge_array,'strain_xx')
    pyv.plot_time_traces(straingauge_array,'strain_yy')
    pyv.plot_time_traces(straingauge_array,'strain_xy')
    plt.show()


if __name__ == "__main__":
    main()