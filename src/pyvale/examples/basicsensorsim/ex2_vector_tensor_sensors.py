# ==============================================================================
# pyvale: the python validation engine
# License: MIT
# Copyright (C) 2025 The Computer Aided Validation Team
# ==============================================================================

"""
Vector and tensor field sensors
================================================================================

TODO
"""
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

# pyvale imports
import pyvale.mooseherder as mh
import pyvale.sensorsim as sens
import pyvale.dataset as dataset

#%%
# 1. Load physics 
# ---------------
# As we did in the last example we load a finite element simulation dataset that 
# comes packaged with `pyvale` in exodus (*.e) format. `pyvale` loads 
# simulations into a `SimData` object  which contains the nodal coordinates, 
# simulation time steps, the nodal physics variables and optionally the element 
# connectivity tables.
#
# We also convert the length units of our simulation from meters to milli-meters
# as our visualisation tools are based on unit scaling by default.

data_path: Path = dataset.mechanical_2d_path()
sim_data: mh.SimData = mh.ExodusLoader(data_path).load_all_sim_data()

disp_keys = ("disp_x","disp_y")
strain_keys = ("strain_xx","strain_yy","strain_xy")

sim_data: mh.SimData  = sens.scale_length_units(scale=1000.0,
                                                sim_data=sim_data,
                                                disp_keys=disp_keys)

#%%
# 2. Create sensor arrays
# -----------------------
# Creating a vector or tensor field sensor array is similar to what we
# have already done for scalar fields we just need to specify the string
# keys for the field components we want to use in the sim data object we have 
# loaded. 

sens_pos: np.ndarray = sens.gen_pos_grid_inside(num_sensors=(2,2,1),
                                                x_lims=(0.0,100.0),
                                                y_lims=(0.0,150.0),
                                                z_lims=(0.0,0.0))

sample_times: np.ndarray = np.linspace(0.0,np.max(sim_data.time),50)

disp_sens_data = sens.SensorData(positions=sens_pos,
                                 sample_times=sample_times)

disp_sens: sens.SensorArrayPoint = sens.SensorFactory.vector_no_errs(
    sim_data,
    disp_sens_data,
    comp_keys=disp_keys,
    spatial_dims=sens.EDim.TWOD,
    descriptor=sens.DescriptorFactory.displacement(),
)

#%%
# For the tensor field sensors we have to separately specify the string keys for
# the normal and deviatoric tensor components.

strain_sens_data = sens.SensorData(positions=sens_pos,
                                   sample_times=sample_times)

strain_sens: sens.SensorArrayPoint = sens.SensorFactory.tensor_no_errs(
    sim_data,
    strain_sens_data,
    norm_comp_keys=("strain_xx","strain_yy",),
    dev_comp_keys=("strain_xy",),
    spatial_dims=sens.EDim.TWOD,
    descriptor=sens.DescriptorFactory.strain(sens.EDim.TWOD),
)


#%%
# 2.1. Add measurement errors
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^
#

pos_offset_xyz = np.array((1.0,1.0,0.0),dtype=np.float64)
pos_offset_xyz = np.tile(pos_offset_xyz,(sens_pos.shape[0],1))

time_offset = np.full((sample_times.shape[0],),0.1)

pos_rand = sens.GenNormal(std=1.0)  # units = mm
time_rand = sens.GenNormal(std=0.1) # units = s

field_err_data = sens.ErrFieldData(
    pos_offset_xyz=pos_offset_xyz,
    time_offset=time_offset,
    pos_rand_xyz=(pos_rand,pos_rand,None),
    time_rand=time_rand
)

disp_err_chain: list[sens.IErrSimulator] = []
disp_err_chain.append(sens.ErrRandNormPercent(std_percent=1.0))
disp_err_chain.append(sens.ErrSysField(disp_sens.get_field(),
                                       field_err_data))

disp_sens.set_error_chain(disp_err_chain)

strain_err_chain: list[sens.IErrSimulator] = []
strain_err_chain.append(sens.ErrRandUnifPercent(low_percent=-1.0,
                                                high_percent=1.0))
strain_err_chain.append(sens.ErrSysField(strain_sens.get_field(),
                                         field_err_data))

strain_sens.set_error_chain(strain_err_chain)


#%%
# 3. Simulate measurements   
# ------------------------
# We run our sensor simulation as normal but we note that the second
# dimension of our measurement array will have either 2 vector components  for 
# the displacement sensors in 2D or 3 tensor components for the strain sensors
# in 2D. 
#
# We also print some of the virtual displacement and strain measurements to 
# the console along with the shapes of the measurement arrays so we can compare
# them. Note that for the tensor sensors the components are ordered as normal
# then deviatoric.

disp_meas: np.ndarray = disp_sens.sim_measurements()
strain_meas: np.ndarray = strain_sens.sim_measurements()

sens_print: int = 0
comp_print: int = 0
time_last: int = 5
time_print = slice(disp_meas.shape[2]-time_last,disp_meas.shape[2])

print(80*"-")
print("DISP. SENSORS")
print(f"The last {time_last} virtual measurements of sensor "
        + f"{sens_print}:\n")

sens.print_measurements(disp_sens,sens_print,comp_print,time_print)

print("\nSTRAIN. SENSORS")
print(f"The last {time_last} virtual measurements of sensor "
        + f"{sens_print}:\n")

sens.print_measurements(strain_sens,sens_print,comp_print,time_print)
print("\n"+80*"-")


#%%
# 4. Visualise simulation & results
# ---------------------------------
# TODO

output_path = Path.cwd() / "pyvale-output"
if not output_path.is_dir():
    output_path.mkdir(parents=True, exist_ok=True)


pv_plot = sens.plot_point_sensors_on_sim(disp_sens,"disp_y")
# Uncomment to show a visualisation of the displacement sensors
# pv_plot.show(cpos="xy")

save_render = output_path / "basic_sensorsim_ex2_disp_locs.svg"
pv_plot.save_graphic(save_render)

pv_plot = sens.plot_point_sensors_on_sim(strain_sens,"strain_yy")
# Uncomment to show a visualisation of the strain sensors
# pv_plot.show(cpos="xy")

save_render = output_path / "basic_sensorsim_ex2_strain_locs.svg"
pv_plot.save_graphic(save_render)

#%%
# We can also plot the traces for each component of
for kk in disp_keys:
    (fig,ax) = sens.plot_time_traces(disp_sens,kk)

    save_traces = output_path/f"basic_sensorsim_ex1_traces_{kk}.svg"
    fig.savefig(save_traces, dpi=300, bbox_inches="tight")
    
for kk in strain_keys:
    (fig,ax) = sens.plot_time_traces(strain_sens,kk)
    
    save_traces = output_path/f"basic_sensorsim_ex1_traces_{kk}.svg"
    fig.savefig(save_traces, dpi=300, bbox_inches="tight")

# Uncomment to show all traces plots
# plt.show()

