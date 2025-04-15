'''
================================================================================
Example: 3d thermocouples on a monoblock

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
    """pyvale example:
    """
    data_path = Path('data/examplesims/monoblock_3d_thermal_out.e')
    sim_data = mh.ExodusReader(data_path).read_all_sim_data()
    field_name = list(sim_data.node_vars.keys())[0] # type: ignore

    # Scale m to mm to make 3D visualisation scaling correct for pyvista
    sim_data = pyv.scale_length_units(1000.0,sim_data)

    use_auto_descriptor = 'manual'
    if use_auto_descriptor == 'factory':
        descriptor = pyvale.SensorDescriptorFactory.temperature_descriptor()
    elif use_auto_descriptor == 'manual':
        descriptor = pyvale.SensorDescriptor()
        descriptor.name = 'Temperature'
        descriptor.symbol = 'T'
        descriptor.units = r'^{\circ}C'
        descriptor.tag = 'TC'
    else:
        descriptor = pyvale.SensorDescriptor()


    field_key = 'temperature'
    t_field = pyvale.FieldScalar(sim_data,
                                 field_key=field_key,
                                 spat_dim=3)

    n_sens = (1,4,1)
    x_lims = (11.5,11.5)
    y_lims = (-11.5,19.5)
    z_lims = (0.0,12.0)
    sens_pos = pyvale.create_sensor_pos_array(n_sens,x_lims,y_lims,z_lims)

    use_sim_time = False
    if use_sim_time:
        sample_times = None
    else:
        sample_times = np.linspace(0.0,np.max(sim_data.time),80)


    tc_array = pyvale.SensorArrayPoint(sens_pos,
                                       t_field,
                                       sample_times,
                                       descriptor)

    errors_on = {'indep_sys': False,
                 'rand': False,
                 'dep_sys': True}

    if errors_on['indep_sys']:
        indep_sys_err1 = pyvale.ErrSysOffset(offset=-5.0)
        indep_sys_err2 = pyvale.ErrSysUniform(low=-10.0,
                                            high=10.0)
        indep_sys_err_int = pyvale.ErrIntegrator([indep_sys_err1,indep_sys_err2],
                                            tc_array.get_measurement_shape())
        tc_array.set_systematic_err_integrator_independent(indep_sys_err_int)

    if errors_on['rand']:
        rand_err1 = pyvale.ErrRandNormPercent(std_percent=5.0)
        rand_err2 = pyvale.ErrRandUnifPercent(low_percent=-5.0,
                                            high_percent=5.0)
        rand_err_int = pyvale.ErrIntegrator([rand_err1,rand_err2],
                                                tc_array.get_measurement_shape())
        tc_array.set_random_err_integrator(rand_err_int)

    if errors_on['dep_sys']:
        dep_sys_err1 = pyvale.ErrSysDigitisation(bits_per_unit=1/20)
        dep_sys_err2 = pyvale.ErrSysSaturation(meas_min=0.0,meas_max=800.0)
        dep_sys_err_int = pyvale.ErrIntegrator([dep_sys_err1,dep_sys_err2],
                                            tc_array.get_measurement_shape())
        tc_array.set_systematic_err_integrator_dependent(dep_sys_err_int)


    measurements = tc_array.get_measurements()
    print(f'\nMeasurements for sensor at top of block:\n{measurements[-1,0,:]}\n')


    (fig,_) = pyvale.plot_time_traces(tc_array,field_key)
    plt.show()

    save_fig = True
    if save_fig:
        save_traces = Path('src/examples/monoblock_thermal_traces_syserrs.png')
        fig.savefig(save_traces, dpi=300, format='png', bbox_inches='tight')



if __name__ == '__main__':
    main()
