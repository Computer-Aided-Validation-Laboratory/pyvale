from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np

# Pyvale imports
import pyvale.sensorsim as sens
import pyvale.mooseherder as mh
import pyvale.dataset as dataset



#%%
# This is a basic set up of thermal data to be plot
# See examples/basics/ex1a_basicscalars_therm2d.py for detail regarding this

data_path = dataset.thermal_2d_path()
sim_data = mh.ExodusReader(data_path).read_all_sim_data()

sim_data = sens.scale_length_units(scale=1000.0,
                                    sim_data=sim_data,
                                    disp_comps=None)

n_sens = (3,2,1)
x_lims = (0.0,100.0)
y_lims = (0.0,50.0)
z_lims = (0.0,0.0)
sens_pos = sens.create_sensor_pos_array(n_sens,x_lims,y_lims,z_lims)


sens_data = sens.SensorData(positions=sens_pos)
field_key: str = "temperature"
tc_array = sens.SensorArrayFactory \
    .thermocouples_basic_errs(sim_data,
                                sens_data,
                                elem_dims=2,
                                field_name=field_key)

sens.animate_trace_with_sensors(tc_array, field_key)


#####################################################

