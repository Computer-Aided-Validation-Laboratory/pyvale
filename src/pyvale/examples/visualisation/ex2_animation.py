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

# NOTE: this functionality is not fully implemented

data_path = dataset.thermal_2d_path()
sim_data = mh.ExodusLoader(data_path).load_all_sim_data()

sim_data = sens.scale_length_units(scale=1000.0,
                                    sim_data=sim_data)

n_sens = (3,2,1)
x_lims = (0.0,100.0)
y_lims = (0.0,50.0)
z_lims = (0.0,0.0)
sens_pos = sens.gen_pos_grid_inside(n_sens,x_lims,y_lims,z_lims)


sens_data = sens.SensorData(positions=sens_pos)
field_key: str = "temperature"


tc_array = sens.SensorFactory.scalar_point(
    sim_data,
    sens_data,
    comp_key=field_key,
    spatial_dims=sens.EDim.TWOD,
    descriptor=sens.DescriptorFactory.temperature(),
)

sens.animate_trace_with_sensors(tc_array, field_key)


#####################################################

