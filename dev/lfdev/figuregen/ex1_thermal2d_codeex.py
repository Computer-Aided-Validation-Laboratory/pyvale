'''
================================================================================
example: thermocouples on a 2d plate

pyvale: the python validation engine
License: MIT
Copyright (C) 2025 The Computer Aided Validation Team
================================================================================
'''
#===============================================================================
# PART 1: Create a sensor array
#===============================================================================

from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pyvale.mooseherder as mh
import pyvale

data_path = Path('src/data/case13_out.e')
data_reader = mh.ExodusLoader(data_path)
sim_data = data_reader.read_all_sim_data()

FIELD_KEY = 'temperature'
temperature_field = pyvale.FieldScalar(sim_data,
                             field_key=FIELD_KEY,
                             spat_dim=2)

n_sens = (4,1,1)
x_lims = (0.0,2.0)
y_lims = (0.0,1.0)
z_lims = (0.0,0.0)
sens_pos = pyvale.create_sensor_pos_array(n_sens,x_lims,y_lims,z_lims)

sample_times = np.linspace(0.0,np.max(sim_data.time),100)

descriptor = pyvale.SensorDescriptorFactory.temperature_descriptor()

thermocouple_array = pyvale.SensorArrayPoint(sens_pos,
                                            temperature_field,
                                            sample_times,
                                            descriptor)

#===============================================================================
# PART 2: Create an error chain
#===============================================================================

# Calculated based on the sensor 'truth' value
indep_sys_err1 = pyvale.ErrSysOffset(offset=-5.0)
indep_sys_err2 = pyvale.ErrSysUnif(low=-10.0,
                                    high=10.0)
indep_sys_err3 = pyvale.ErrSysPositionRand(temperature_field,
                                            sens_pos,
                                            (0.05,0.05,None),
                                            sample_times)
indep_sys_err_int = pyvale.ErrIntegrator([indep_sys_err1,
                                            indep_sys_err2,
                                            indep_sys_err3],
                                    thermocouple_array.get_measurement_shape())
thermocouple_array.set_systematic_err_integrator_independent(indep_sys_err_int)

# Calculated based on the sensor 'truth' value
rand_err1 = pyvale.ErrRandNormPercent(std_percent=5.0)
rand_err_int = pyvale.ErrIntegrator([rand_err1,],
                                    thermocouple_array.get_measurement_shape())
thermocouple_array.set_random_err_integrator(rand_err_int)

# Calculated based on the accumulated error in the error chain
dep_sys_err1 = pyvale.ErrSysDigitisation(bits_per_unit=1/10)
dep_sys_err2 = pyvale.ErrSysSaturation(meas_min=0.0,meas_max=300.0)
dep_sys_err_int = pyvale.ErrIntegrator([dep_sys_err1,dep_sys_err2],
                                    thermocouple_array.get_measurement_shape())
thermocouple_array.set_systematic_err_integrator_dependent(dep_sys_err_int)

measurements = thermocouple_array.calc_measurements()

#===============================================================================
# PART 3: Run virtual experiments
#===============================================================================

NUM_EXPERIMENTS = 3
save_figs = True
for ee in range(NUM_EXPERIMENTS):
    thermocouple_array.calc_measurements()

    pyvale.print_measurements(thermocouple_array,
                            (0,1), # Sensor 1
                            (0,1), # Component 1: scalar field = 1 component
                            (measurements.shape[2]-5,measurements.shape[2]))

    (fig,_) = pyvale.plot_time_traces(thermocouple_array,FIELD_KEY)


# Calling 'get' does not resample the errors
measurements = thermocouple_array.get_measurements()

pyvale.print_measurements(thermocouple_array,
                        (0,1), # Sensor 1
                        (0,1), # Component 1: scalar field = 1 component
                        (measurements.shape[2]-5,measurements.shape[2]))

(fig,_) = pyvale.plot_time_traces(thermocouple_array,FIELD_KEY)
plt.show()

pv_plot = pyvale.plot_point_sensors_on_sim(thermocouple_array,FIELD_KEY)
pv_plot.camera_position = [(-0.295, 1.235, 3.369),
                            (1.0274, 0.314, 0.0211),
                            (0.081, 0.969, -0.234)]
pv_plot.show(cpos="xy")

#===============================================================================
# PART 4: Visualise Sensor Positions
#===============================================================================

save_render = Path('src/examples/figuregen/codex_sensvis.svg')
pv_plot.save_graphic(save_render) # only for .svg .eps .ps .pdf .tex
pv_plot.screenshot(save_render.with_suffix('.png'))


#===============================================================================
# SAVE FIGURES
#===============================================================================

# Calling 'calc' samples all of the errors generating a new experiment
NUM_EXPERIMENTS = 3
save_figs = True
for ee in range(NUM_EXPERIMENTS):
    thermocouple_array.calc_measurements()

    pyvale.print_measurements(thermocouple_array,
                            (0,1), # Sensor 1
                            (0,1), # Component 1: scalar field = 1 component
                            (measurements.shape[2]-5,measurements.shape[2]))

    (fig,_) = pyvale.plot_time_traces(thermocouple_array,FIELD_KEY)

    save_traces = Path('src/examples/figuregen/'+
                        f'codex_traces_exp{ee+1}.png')
    fig.savefig(save_traces, dpi=300, format='png', bbox_inches='tight')

# Calling 'get' does not resample the errors
measurements = thermocouple_array.get_measurements()

pyvale.print_measurements(thermocouple_array,
                        (0,1), # Sensor 1
                        (0,1), # Component 1: scalar field = 1 component
                        (measurements.shape[2]-5,measurements.shape[2]))

(fig,_) = pyvale.plot_time_traces(thermocouple_array,FIELD_KEY)

save_traces = Path('src/examples/figuregen/'+
                  f'codex_traces_exp0.png')
fig.savefig(save_traces, dpi=300, format='png', bbox_inches='tight')







